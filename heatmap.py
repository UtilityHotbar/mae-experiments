import torch
import torch.nn as nn

from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import accuracy

import models_vit

import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import PIL
import os
import datetime
import pickle
import time
import random

# Experiment params
DATA_PATH = 'heatmap_images'
LOG_DIR = 'logs'
OUTPUT_DIR = 'heatmap_results'
FIXED_SEED = None
DEVICE = 'cpu'
REPORT_INTERVAL_SIZE = 16

GET_BASE_RESULT = True

# Model params (just the defaults basically)
MODEL_NAME = 'vit_base_patch16'
BATCH_SIZE = 16
CLASSES = 1000
DROP_PATH_RATE = 0.1
GLOBAL_POOL = True
NUM_WORKERS = 8
PIN_MEM = True
ACCUM_ITER = 1
WEIGHT_DECAY = 0.05
LAYER_DECAY = 0.75
INPUT_SIZE = 224

BASE_LEARNING_RATE = 1e-3
LEARNING_RATE = BASE_LEARNING_RATE

HEATMAP = {}

# Mocking up an arguments object to feed the model path to the model loader
class ResumeObject():
    def __init__(self) -> None:
        self.resume = 'mae_finetuned_vit_base.pth'

RESUME = ResumeObject()

def custom_evaluate(data_loader, model, device, base_pos_embed):

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, REPORT_INTERVAL_SIZE, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        heat_map = {}
        for i in range(167):
            print(f'Cutting {i}')

            if GET_BASE_RESULT:
                clen = 0
            else:
                model.pos_embed = nn.Parameter(torch.cat((base_pos_embed[:,:i+1], base_pos_embed[:,i+1+1:]), 1))
                clen = 1
            # compute output
            with torch.cuda.amp.autocast():
                output = model(images, i, clen)
                # print(output.shape)

                loss = criterion(output, target)

                heat_map[i] = {}
                heat_map[i]['raw'] = output.detach()

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                heat_map[i]['acc1'] = acc1
                heat_map[i]['acc5'] = acc5
                batch_size = images.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
                # gather the stats from all processes
                metric_logger.synchronize_between_processes()
                print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                    .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
            if GET_BASE_RESULT:
                break
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, heat_map

def build_experiment_dataset(data_path, input_size=INPUT_SIZE):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    eval_transform = transforms.Compose(t)

    root = os.path.join(data_path, 'val')
    dataset = datasets.ImageFolder(root, transform=eval_transform)

    print(dataset)

    return dataset


def main():
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    device = torch.device(DEVICE)

    if FIXED_SEED:
        torch.manual_seed(FIXED_SEED)
        np.random.seed(FIXED_SEED)
    dataset_val = build_experiment_dataset(DATA_PATH)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        drop_last=False
    )

    model = models_vit.__dict__[MODEL_NAME](
        num_classes=CLASSES,
        drop_path_rate=DROP_PATH_RATE,
        global_pool=GLOBAL_POOL,
    )

    # We change the patching process to cut out a patch
    # model.foward_features = forward_features

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = BATCH_SIZE * ACCUM_ITER * 1

    print("base lr: %.2e" % (LEARNING_RATE * 256 / eff_batch_size))
    print("actual lr: %.2e" % LEARNING_RATE)

    print("accumulate grad iterations: %d" % ACCUM_ITER)
    print("effective batch size: %d" % eff_batch_size)

    param_groups = lrd.param_groups_lrd(model_without_ddp, WEIGHT_DECAY,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=LAYER_DECAY
    )
    optimizer = torch.optim.AdamW(param_groups, lr=LEARNING_RATE)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    misc.load_model(args=RESUME, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    base_pos_embed = model.pos_embed.detach().clone()
    test_stats, results = custom_evaluate(data_loader_val, model, device, base_pos_embed)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    if GET_BASE_RESULT:
        with open(OUTPUT_DIR+'/base_results.pickle', 'wb') as g:
            pickle.dump(results, g)
    else:
        with open(OUTPUT_DIR+'/results.pickle', 'wb') as g:
            pickle.dump(results, g)
    
    exit(0)

if __name__ == '__main__':
    main()