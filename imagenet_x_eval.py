# imports
import matplotlib.pyplot as plt
import pandas as pd
import torch

from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from imagenet_x.evaluate import ImageNetX, get_vanilla_transform
from imagenet_x import FACTORS, plots

import models_vit
import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import copy

# Declare dataset
imagenet_val_path = 'imagenet_valid_2012/'
transforms = get_vanilla_transform()
dataset = ImageNetX(imagenet_val_path, transform=transforms)

# Load Resnet model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
device = 'cpu'
batch_size = 16
num_workers = 2

model.eval()
model.to(device)

# Load MAE-VIT model

BATCH_SIZE = batch_size
BASE_LEARNING_RATE = 1e-3
LEARNING_RATE = BASE_LEARNING_RATE
WEIGHT_DECAY = 0.05
LAYER_DECAY = 0.75
class ResumeObject():
    def __init__(self) -> None:
        self.resume = 'mae_finetuned_vit_base.pth'

RESUME = ResumeObject()

other_model = models_vit.__dict__['vit_base_patch16'](
        num_classes=1000,
        drop_path_rate=0.1,
        global_pool=True,
    )
other_model.eval()
other_model.to(device)

model_without_ddp = other_model
n_parameters = sum(p.numel() for p in other_model.parameters() if p.requires_grad)

n_parameters = sum(p.numel() for p in other_model.parameters() if p.requires_grad)

print("Model = %s" % str(model_without_ddp))
print('number of params (M): %.2f' % (n_parameters / 1.e6))

eff_batch_size = BATCH_SIZE

print("base lr: %.2e" % (LEARNING_RATE * 256 / eff_batch_size))
print("actual lr: %.2e" % LEARNING_RATE)

print("accumulate grad iterations: %d" % 1)
print("effective batch size: %d" % eff_batch_size)

param_groups = lrd.param_groups_lrd(model_without_ddp, WEIGHT_DECAY,
    no_weight_decay_list=model_without_ddp.no_weight_decay(),
    layer_decay=LAYER_DECAY
)
optimizer = torch.optim.AdamW(param_groups, lr=LEARNING_RATE)
loss_scaler = NativeScaler()

criterion = torch.nn.CrossEntropyLoss()

misc.load_model(args=RESUME, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

# Evaluate model on ImageNetX using simple loop

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
)

print('Now evaluating.')
correct = 0
total = 0

other_correct = 0
other_total = 0

try:
    with torch.no_grad():
        for data, target, annotations in tqdm(loader, desc="Evaluating on Imagenet-X"):
            other_annotations = copy.deepcopy(annotations)
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            mask = pred.eq(target.view_as(pred))
            correct += annotations[mask,:].to(dtype=torch.int).sum(dim=0)
            total += annotations.to(dtype=torch.int).sum(dim=0)

            other_output = other_model(data, 0, 0)
            other_pred = other_output.argmax(dim=1)
            other_mask = other_pred.eq(target.view_as(other_pred))
            other_correct += other_annotations[other_mask,:].to(dtype=torch.int).sum(dim=0)
            other_total += other_annotations.to(dtype=torch.int).sum(dim=0)
finally:
    # Compute accuracies per factor for resnet
    factor_accs = (correct/total).cpu().detach().numpy()
    results = pd.DataFrame({'Factor': FACTORS, 'acc': factor_accs}).sort_values('acc', ascending=False)

    # Compute error ratios per factor
    results['Error ratio'] = (1 - results['acc']) / (1-(correct.sum()/total.sum()).item())

    # Plot results
    # plots.plot_bar_plot(results, x='Factor', y='Error ratio')
    # plt.show()
    print('===RESNET==')
    print(results)
    results.to_csv('output/imagenet_x_resnet.csv')


    # Compute accuracies per factor for other model
    other_factor_accs = (other_correct/other_total).cpu().detach().numpy()
    other_results = pd.DataFrame({'Factor': FACTORS, 'acc': other_factor_accs}).sort_values('acc', ascending=False)

    # Compute error ratios per factor for other model
    other_results['Error ratio'] = (1 - other_results['acc']) / (1-(other_correct.sum()/other_total.sum()).item())

    # Plot results
    # plots.plot_bar_plot(other_results, x='Factor', y='Error ratio')
    # plt.show()
    print('===MAE-VIT===')
    print(other_results)
    other_results.to_csv('output/imagenet_x_mae_vit.csv')
