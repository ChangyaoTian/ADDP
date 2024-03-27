
import math
import numpy as np
import os

import torch
import torch.nn.functional as F

import util.misc as misc
from modeling.utils import random_masking_mae, save_imgs
from modeling.encoders.models_addp import MaskedGenerativeEncoderViT


def mask_by_random_topk(mask_len, probs, temperature=1.0):
    masks = torch.zeros_like(probs).bool()
    confidence = torch.log(probs) + torch.Tensor(temperature * np.random.gumbel(size=probs.shape)).cuda()
    topk_probs, topk_idxs = torch.topk(confidence, k=mask_len, dim=-1, largest=False)
    row_idxs = torch.arange(0, masks.shape[0], device=masks.device).unsqueeze(-1)
    masks[row_idxs, topk_idxs] = True
    return masks


def nucleus_sampling(logits, top_p=1.0):
    # Get probabilities $P(x_i | x_{1:i-1})$
    probs = F.softmax(logits, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)

    # Get the cumulative sum of probabilities in the sorted order
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the cumulative sums less than $p$.
    nucleus = cum_sum_probs < top_p

    # Prepend ones so that we add one token after the minimum number
    # of tokens with cumulative probability less that $p$.
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)

    # Get log probabilities and mask out the non-nucleus
    sorted_log_probs = torch.log(sorted_probs)
    sorted_log_probs[~nucleus] = float('-inf')

    # Sample from the sampler
    sampled_sorted_indexes = torch.distributions.categorical.Categorical(logits=sorted_log_probs).sample()
    # sampled_sorted_indexes = self.sampler(sorted_log_probs)

    # Get the actual indexes
    res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))

    #
    return res.squeeze(-1)

@torch.no_grad()
def masked_generate_multi_step_addp(model, data_loader_val, device, args):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = '[EVAL] multi-step generation: '
    print_freq = 5
    model: MaskedGenerativeEncoderViT = model.module

    num_iter = args.num_iteration
    choice_temperature = args.choice_temperature
    mask_token_id = 10000
    num_patches = 256
    codebook_emb_dim = 256
    mask_token_idxs = mask_token_id * torch.ones(1, num_patches, device=device)

    args.save_dir_suffix = args.save_dir_suffix + f'_n{num_iter}_t{choice_temperature:.1f}'

    for i, data in enumerate(metric_logger.log_every(data_loader_val, print_freq, header)):
        imgs, labels, path = data
        imgs = imgs.to(device)
        initial_token_idxs = mask_token_idxs.expand(imgs.shape[0], -1).unsqueeze(-1)
        masks, _ = random_masking_mae(initial_token_idxs, args.mask_ratio)
        
        gt_indices = model.vq_encode(imgs)[0].detach().long()
        Tt_bar = gt_indices.clone()
        xt = imgs

        for step in range(num_iter):
            # predict token distribution from pixels
            Tt_next_logits, token_all_mask = predict_token_from_pixel(model, xt, Tt_bar, masks)

            if args.sampling_strategy == "top_p":
                sampled_ids = nucleus_sampling(Tt_next_logits, top_p=args.top_p)
            else:
                sampled_ids = torch.distributions.categorical.Categorical(logits=Tt_next_logits).sample()
            Tt_next_bar = torch.where(masks.bool(), sampled_ids, Tt_bar)

            # compute masks_next
            masks_next = compute_mask_next(Tt_next_logits, Tt_next_bar, token_all_mask, step, num_iter, args)

            def compute_x_next(Tt_next_logits, T_bar, mask):
                # iteratively compute xt_next_prime
                Tt_hat_next = F.softmax(Tt_next_logits, dim=-1)

                # merge soft and hard embedding
                soft_embedding = Tt_hat_next @ model.vqgan.quantize.embedding.weight
                hard_embedding = model.vqgan.quantize.get_codebook_entry(T_bar, shape=(T_bar.shape[0], 16, 16, codebook_emb_dim))
                B, L = mask.shape
                embedding_mask = mask.view(B, 1, int(L**.5), int(L**.5))
                soft_embedding = soft_embedding.reshape(B, int(L**.5), int(L**.5), -1).permute(0, 3, 1, 2)
                
                z_q = torch.where(embedding_mask.bool(), soft_embedding.float(), hard_embedding)
                
                # generate pixel
                xt_next = model.vqgan.decode(z_q)
                return xt_next

            xt_next = compute_x_next(Tt_next_logits, T_bar=Tt_next_bar, mask=masks_next)
            
            xt = xt_next
            Tt_bar = Tt_next_bar
            masks = masks_next
        predicted_imgs = xt
        save_imgs(predicted_imgs, path, args=args, subdir='predicted_img')


def predict_token_from_pixel(model: MaskedGenerativeEncoderViT, xt, Tt_bar, masks):
    latent, token_all_mask = model.generate_mask_mlp_tokens(xt, token_all_mask=masks)
    latent = model.forward_online_encoder(latent)
    # decoder
    logits = model.forward_decoder(latent, Tt_bar, token_all_mask)

    codebook_size = 1024
    num_patches = 256
    token_all_mask  = token_all_mask[:,  :1+num_patches]
    Tt_next_logits = logits[:, 1:1+num_patches, :codebook_size]

    return Tt_next_logits, token_all_mask


def compute_mask_next(logits, Tt_next_bar, token_all_mask, step, num_iter, args):
    _CONFIDENCE_OF_KNOWN_TOKENS = +np.inf
    num_patches = 256

    # get sampled probability
    probs = F.softmax(logits, dim=-1)
    sampled_probs = torch.squeeze(
        torch.gather(probs, dim=-1, index=torch.unsqueeze(Tt_next_bar, -1)), -1)
    sampled_probs = torch.where(token_all_mask[:, 1:].bool(), sampled_probs.double(), _CONFIDENCE_OF_KNOWN_TOKENS).float()

    # compute length of mask for next timestep
    step_ratio = 1. * (step + 1) / num_iter
    if args.mask_ratio_sample_strategy == 'cosine':
        mask_ratio = math.cos(math.pi / 2. * step_ratio)
    elif args.mask_ratio_sample_strategy == 'linear':
        mask_ratio = 1. - step_ratio
    
    mask_len = math.ceil(num_patches * mask_ratio * args.mask_ratio)
    mask_len = np.clip(mask_len, 1, int(num_patches * args.mask_ratio))

    if args.temperature_strategy == "static":
        temp = args.choice_temperature
    else:
        temp = args.choice_temperature * (1 - step_ratio)
    masks_next = mask_by_random_topk(mask_len, sampled_probs, temp)

    return masks_next
