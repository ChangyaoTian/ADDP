import argparse
import numpy as np
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DistributedSampler, DataLoader

import util.misc as misc
from util.datasets import ImagenetDataset

import modeling

from engine_generate import masked_generate_multi_step_addp


def get_args_parser():
    parser = argparse.ArgumentParser('ADDP Un-conditional Image Generation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # no optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')


    # generation parameters
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--save_dir_suffix', type=str, default='')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # additional parameters
    parser.add_argument('--auto_resume', default=True)

    parser.add_argument('--drop_path_rate', default=0.0, type=float)
    parser.add_argument('--drop_out_rate', default=0.1, type=float)
    parser.add_argument('--clip_grad', default=None, type=float)
    parser.add_argument('--init_values', default=None, type=float)

    parser.add_argument('--fp32', default=False, action='store_true')

    parser.add_argument('--attn_drop', default=0.0, type=float)
    
    parser.add_argument('--amp_growth_interval', default=2000, type=int)

    parser.add_argument('--port', default=28507, type=int)
    

    # MAGE params
    parser.add_argument('--mask_ratio_min', type=float, default=0.5,
                        help='Minimum mask ratio')
    parser.add_argument('--mask_ratio_max', type=float, default=1.0,
                        help='Maximum mask ratio')
    parser.add_argument('--mask_ratio_mu', type=float, default=0.55,
                        help='Mask ratio distribution peak')
    parser.add_argument('--mask_ratio_std', type=float, default=0.25,
                        help='Mask ratio distribution std')

    # fp32
    parser.add_argument('--attn_fp32', default=False, action='store_true')

    # generate option
    parser.add_argument('--num_iteration', default=5, type=int)
    parser.add_argument('--token_predictor_ckpt', type=str, default=None)
    parser.add_argument('--token_predictor_name', type=str, default='mage_vit_base_patch16')
    parser.add_argument('--decoder_depth', default=8, type=int)

    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('--mask_ratio_sample_strategy', default='cosine', choices=['cosine', 'linear'])
    parser.add_argument('--choice_temperature', default=6.0, type=float)
    parser.add_argument('--temperature_strategy', default='dynamic', choices=['dynamic', 'static'])
    parser.add_argument('--sampling_strategy', default='default', choices=['default', 'top_p'])
    parser.add_argument('--top_p', default=1.0, type=float)
    
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # build dataloader
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor()]
    )
    dataset_val = ImagenetDataset('val',
                                    args.data_path,
                                    transform=transform_val,
                                )
    # dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_val = DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_val)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # build model
    model = modeling.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, args=args)

    checkpoint = torch.load(args.pretrain, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.pretrain)
    checkpoint_model = checkpoint['model']

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print(f"Start eval")
    masked_generate_multi_step_addp(model, data_loader_val, device=device, args=args)
    dist.barrier()
    print(f"All finished")
    save_dir = os.path.dirname(args.pretrain)
    ckpt_name = args.pretrain.split('/')[-1].split('.')[0]
    save_root_path = os.path.join(save_dir, ckpt_name + args.save_dir_suffix, 'predicted_img')
    print("image save path:", save_root_path)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
