import math
import os
import sys
from typing import Iterable
from pathlib import Path

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if misc.is_main_process():
        collapse_file = os.path.join(args.output_dir, 'collapse')
        if os.path.isfile(collapse_file):
            os.system(f"rm -f {collapse_file}")

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = data[0]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(not args.fp32)):
            loss, outputs = model(samples)
            metric_logger.update(**outputs)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            # print("Loss is {}, stopping training".format(loss_value))
            # sys.exit(1)
            if args.fp32:
                print("Loss is {} in fp32/tf32 mode, exit".format(loss_value))
                collapse_file = os.path.join(args.output_dir, 'collapse')
                Path(collapse_file).touch()
                sys.exit(1)
            else:
                print("Warning: Loss is {}, but not stopping training".format(loss_value))

        loss = loss / accum_iter
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0, clip_grad=args.clip_grad)
        if args.fp32:
            loss_scale = None
        else:
            loss_scale = loss_scaler.state_dict()['scale']

        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(loss_scale=loss_scale)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        outputs_reduced = {k_: misc.all_reduce_mean(v_) for k_, v_ in outputs.items()}
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('grad_norm', grad_norm, epoch_1000x)
            if loss_scale is not None:
                log_writer.add_scalar('loss_scale', loss_scale, epoch_1000x)
            for k_, v_ in outputs_reduced.items():
                log_writer.add_scalar(f'train/{k_}', v_, epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
