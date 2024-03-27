import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from multiprocessing import Pool
import time
import random
import cv2
import scipy.stats as stats
import math

def random_masking_mae(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore


def save_img(img_path):
    img, path = img_path
    path = path.replace("JPEG", "png")
    flag = cv2.imwrite(path, img)
    i = 0
    while flag is False and i < 10:
        print(f"No.{i} failed to write image to", path, force=True)
        time.sleep(random.randint(1,5))
        i += 1
        flag = cv2.imwrite(path, img)
    if i == 10 and not flag:
        print("failed to write image to", path, force=True)


def save_imgs(imgs, paths, args, subdir='predicted_img'):
    save_dir = os.path.dirname(args.pretrain)
    ckpt_name = args.pretrain.split('/')[-1].split('.')[0]
    save_root_path = os.path.join(save_dir, ckpt_name + args.save_dir_suffix, subdir)
    os.makedirs(save_root_path, exist_ok=True)
    imgs = torch.clamp(imgs.permute(0,2,3,1) * 255, 0, 255)
    imgs = imgs.to(dtype=torch.uint8, device='cpu').numpy()[:,:,:,::-1]
    # save img
    for img_ori, path in zip(imgs, paths):
        save_path = os.path.join(save_root_path, path)
        save_img((img_ori, save_path))


def compute_mask_ratio_probs(mask_ratio_min, mask_ratio_max, mask_ratio_mu, mask_ratio_std, num_iter=100):
    mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)
    step_ratios = 1. * (torch.arange(num_iter + 1)) / num_iter # [1,T]
    mask_ratios = torch.clamp(torch.cos(math.pi / 2. * step_ratios), 0., 1.) # (0,1]
    pdfs = []
    mask_rs = []
    for mask_ratio in mask_ratios:
        if mask_ratio >= mask_ratio_min and mask_ratio < mask_ratio_max:
            pdf = mask_ratio_generator.pdf(mask_ratio)
            pdfs.append(pdf)
            mask_rs.append(mask_ratio)
    mask_rs.reverse()
    pdfs.reverse()
    pdfs = torch.tensor(pdfs)
    mask_rs = torch.tensor(mask_rs)
    probs = pdfs / pdfs.sum()     
    return probs, mask_rs

