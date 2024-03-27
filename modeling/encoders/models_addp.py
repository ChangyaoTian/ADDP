from functools import partial
import math
import random

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp

from omegaconf import OmegaConf
from image_synthesis.taming.models.vqgan import VQModel
import numpy as np
from modeling.encoders.token_predictor import token_predictor
from modeling.utils import compute_mask_ratio_probs


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        with torch.cuda.amp.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale

        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            _, attn = self.attn(self.norm1(x))
            return attn
        else:
            y, _ = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size, last_fc=False):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        if last_fc:
            self.fc2 = nn.Linear(word_emb_dim, vocab_size)
        else:
            self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)

        if word_embeddings is None:
            logits = self.fc2(mlm_hidden)
        else:
            word_embeddings = word_embeddings.transpose(0, 1)
            logits = torch.matmul(mlm_hidden, word_embeddings)
            logits = logits + self.bias
        return logits


class MaskedGenerativeEncoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 vqgan_ckpt_path='./exp/pretrained_model/vqgan_jax_strongaug.ckpt',
                 vqgan_cfg_path='./configs/release/vqgan.yaml',
                 args=None,
                ):
        super().__init__()
        self.args = args
        # --------------------------------------------------------------------------
        # VQGAN specifics
        config = OmegaConf.load(vqgan_cfg_path).model
        self.vqgan = VQModel(ddconfig=config.params.ddconfig,
                            n_embed=config.params.n_embed,
                            embed_dim=config.params.embed_dim,
                            ckpt_path=vqgan_ckpt_path)
        for param in self.vqgan.parameters():
            param.requires_grad = False

        codebook_size = config.params.n_embed
        vocab_size = codebook_size + 1000 + 1  # 1024 codebook size, 1000 classes, 1 for mask token.
        self.fake_class_label = codebook_size + 1100 - 1024
        self.mask_token_label = vocab_size - 1
        self.token_emb = nn.Embedding(vocab_size, embed_dim)

        self.mask_ratio_min = mask_ratio_min
        probs, mask_rs = compute_mask_ratio_probs(mask_ratio_min, mask_ratio_max, mask_ratio_mu, mask_ratio_std, num_iter=100)
        self.mask_ratio_generator = torch.distributions.Categorical(probs=probs)
        self.mask_ratios = mask_rs

        # --------------------------------------------------------------------------
        # ADDP Encoder specifics
        dropout_rate = args.drop_out_rate
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.tokenizer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.tokenizer_dropout = nn.Dropout(0.1)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # ADDP Decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))  # learnable pos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.pos_token_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  
        self.decoder_pos_token_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))  # learnable pos embedding
        self.pos_mask_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  
        self.decoder_pos_mask_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  
        
        # --------------------------------------------------------------------------
        # ADDP loss specifics
        self.mlm_layer = MlmLayer(
            feat_emb_dim=decoder_embed_dim,
            word_emb_dim=embed_dim,
            vocab_size=vocab_size,
            last_fc=False
        )

        self.norm_pix_loss = norm_pix_loss

        self.criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smoothing)

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # ADDP off-the-shelf token predictor specifics
        self.token_predictor = token_predictor(
            model_name=args.token_predictor_name,
            norm_pix_loss=False,
            vqgan=self.vqgan
        )
        checkpoint = torch.load(self.args.token_predictor_ckpt, map_location='cpu')
        msg = self.token_predictor.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        print("Resume token_predictor checkpoint %s" % self.args.token_predictor_ckpt)
        for param in self.token_predictor.parameters():
            param.requires_grad = False
        self.token_predictor.eval()


    def initialize_weights(self):
        # initialization
        def init_pos_embed(pos_embed):
            torch.nn.init.normal_(pos_embed, std=.02)

        init_pos_embed(self.pos_embed)
        init_pos_embed(self.decoder_pos_embed)
        init_pos_embed(self.decoder_pos_embed_learned)
        init_pos_embed(self.pos_token_embed)
        init_pos_embed(self.decoder_pos_token_embed)
        init_pos_embed(self.pos_mask_embed)
        init_pos_embed(self.decoder_pos_mask_embed)
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.token_emb.weight, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def vq_encode(self, x):
        z_q, _, token_tuple = self.vqgan.encode(x)
        _, _, token_indices = token_tuple
        token_indices = token_indices.reshape(z_q.size(0), -1)
        return token_indices, z_q

    def random_masking(self, token_indices):
        # masking
        bsz, seq_len = token_indices.size()
        mask_rate_idx = self.mask_ratio_generator.sample(torch.Size([1]))
        mask_rate = self.mask_ratios[mask_rate_idx]
        mask_rate_idx_offset = random.randint(1, 5)
        next_mask_rate_idx = min(mask_rate_idx + mask_rate_idx_offset, len(self.mask_ratios)-1)
        next_mask_rate = self.mask_ratios[next_mask_rate_idx]

        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        num_next_masked_tokens = int(np.ceil(seq_len * next_mask_rate))

        # it is possible that two elements of the noise is the same, so do a while loop to avoid it
        while True:
            noise = torch.rand(bsz, seq_len, device=token_indices.device)  # noise in [0, 1]
            sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
            cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
            cutoff_next_mask = sorted_noise[:, num_next_masked_tokens-1:num_next_masked_tokens]
            token_all_mask = (noise <= cutoff_mask).float()
            token_all_next_mask = (noise <= cutoff_next_mask).float()
            if token_all_mask.sum() == bsz*num_masked_tokens and \
                    token_all_next_mask.sum() == bsz*num_next_masked_tokens:
                break
            else:
                print("Rerandom the noise!")
        
        return token_all_mask, token_all_next_mask

    def normalize_input(self, x):
        x = torch.clamp(x, 0., 1.)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(x)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(x)
        x = (x - mean) / std
        return x
    
    def generate_mask_mlp_tokens(self, x, token_all_mask):
        x = self.normalize_input(x)
        input_embeddings = self.patch_embed(x)

        input_embeddings = input_embeddings + self.pos_embed[:, 1:, :]

        # add cls token
        cls_tokens = self.cls_token.expand(input_embeddings.shape[0], -1, -1) + self.pos_embed[:, 0, :].unsqueeze(1)
        input_embeddings = torch.cat((cls_tokens, input_embeddings), dim=1)

        input_embeddings = self.tokenizer_norm(input_embeddings)
        input_embeddings = self.tokenizer_dropout(input_embeddings)

        # no mask for cls token
        token_all_mask  = torch.cat([torch.zeros(x.shape[0], 1).cuda(), token_all_mask], dim=1)

        return input_embeddings, token_all_mask

    def forward_vqgan(self, x):
        token_indices_z0, z0 = self.vq_encode(x)
        # mask sampling
        token_mask_t, token_mask_t_next = self.random_masking(token_indices_z0)

        # Diffusion process
        z_t_pred, z_t_prev_prob_target = self.forward_token_predictor(
            token_indices_z0, token_mask_t, token_mask_t_next
        )

        # Token-to-Pixel Decoding
        B, L = token_mask_t.shape
        embedding_mask = token_mask_t.view(B, 1, int(L**.5), int(L**.5))
        z_t = torch.where(embedding_mask.bool(), z_t_pred.float(), z0)
        with torch.no_grad():
            x_t = self.vqgan.decode(z_t)

        return x_t, token_indices_z0, z_t_prev_prob_target, token_mask_t

    def forward_token_predictor(self, token_indices_z0, token_mask_t, token_mask_t_next):
        # forward twice
        z_t_pred = self.token_predictor.sample_tokens(
                    token_indices_z0, token_mask_t_next,
                    return_pred_tokens=True, codebook=self.vqgan.quantize.embedding.weight,
                    )
        z_t_prev_prob_target = self.token_predictor.sample_tokens(
                    token_indices_z0, token_mask_t,
                    return_pred_tokens=False, codebook=self.vqgan.quantize.embedding.weight,
                    )
        return z_t_pred, z_t_prev_prob_target

    def forward_encoder(self, x):
        x_t, token_indices_z0, z_t_prev_prob_target, token_mask_t = self.forward_vqgan(x)
        
        latent, token_mask_t = self.generate_mask_mlp_tokens(x_t, token_all_mask=token_mask_t)
        # apply Transformer blocks
        e_t = self.forward_online_encoder(latent)

        return e_t, token_indices_z0, z_t_prev_prob_target, token_mask_t

    def forward_online_encoder(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_decoder(self, x, token_indices, token_all_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.expand(token_all_mask.shape[0], token_all_mask.shape[1], -1)

        # put undropped tokens into original sequence
        cls_mask_tokens = self.mask_token.expand(token_all_mask.shape[0], 1, -1)
        x = torch.cat([cls_mask_tokens, x[:, 1:]], dim=1)
        # set undropped but masked positions with mask
        x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x)
        
        input_token_embeddings = self.token_emb(token_indices)
        input_token_embeddings = input_token_embeddings[~token_all_mask[:, 1:].bool()].reshape(input_token_embeddings.shape[0], 
                                                                                        -1, input_token_embeddings.shape[-1])
        x_after_pad = torch.cat([x_after_pad, input_token_embeddings], dim=1)

        # compute pos embed
        decoder_pos_embed_learned = self.decoder_pos_embed_learned.expand(x.shape[0], -1, -1)
        decoder_mask_pos_embed = self.decoder_pos_mask_embed.expand(x.shape[0], -1, -1)
        decoder_pos_embed_learned = torch.where(token_all_mask[:,:decoder_pos_embed_learned.shape[1]].unsqueeze(-1).bool(), 
                                                    decoder_mask_pos_embed, decoder_pos_embed_learned)
        decoder_token_pos_embed_learned = decoder_pos_embed_learned
        decoder_token_pos_embed_learned = self.decoder_pos_token_embed.expand(x.shape[0], -1, -1)
        decoder_token_pos_embed_learned = decoder_token_pos_embed_learned[:,1:,:][~token_all_mask[:,1:decoder_pos_embed_learned.shape[1]].bool()]
        decoder_token_pos_embed_learned = decoder_token_pos_embed_learned.reshape(x.shape[0], -1, x.shape[-1])
        decoder_pos_embed_learned = torch.cat([decoder_pos_embed_learned, decoder_token_pos_embed_learned], dim=1)
        # add pos embed
        x = x_after_pad + decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        word_embeddings = self.token_emb.weight.data.detach()
        out = self.mlm_layer(x, word_embeddings)
        return out
    
    def forward_loss(self, softmax_prob, logits, mask):
        bsz, seq_len, vocab_size = logits.size()
        _, codebook_size, h, w = softmax_prob.size()

        concat_zero_prob = torch.zeros((bsz, vocab_size-codebook_size, h, w)).to(softmax_prob)
        softmax_prob = torch.cat([softmax_prob, concat_zero_prob], dim=1)
        logits = logits[:, 1:1+h*w].reshape(bsz, h, w, -1).permute(0, 3, 1, 2)
        # logits and mask are with seq_len+1 but gt_indices is with seq_len
        loss = self.criterion(logits, softmax_prob)

        loss = loss.reshape(bsz, h*w)
        loss = (loss * mask[:, 1:1+h*w]).sum() / mask[:, 1:1+h*w].sum()
        return loss

    def forward(self, x1):
        et, token_indices_z0, z_t_prev_prob_target, token_mask_t = self.forward_encoder(x1)
        z_t_prev_logits = self.forward_decoder(et, token_indices_z0, token_mask_t)
        loss = self.forward_loss(z_t_prev_prob_target, z_t_prev_logits, token_mask_t)
        outputs = {'loss': loss.item()}

        return loss, outputs


def addp_vit_base_patch16(**kwargs):
    args = kwargs.pop('args', None)
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=args.decoder_depth, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
        mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
        args=args,
        **kwargs
    )
    return model


def addp_vit_large_patch16(**kwargs):
    args = kwargs.pop('args', None)
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=args.decoder_depth, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
        mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
        args=args,
        **kwargs
    )
    return model
