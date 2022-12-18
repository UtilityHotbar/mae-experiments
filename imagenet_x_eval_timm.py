# imports
import matplotlib.pyplot as plt
import pandas as pd
import torch

from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from imagenet_x.evaluate import ImageNetX, get_vanilla_transform
from imagenet_x import FACTORS, plots

import copy
import timm

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

# Load timm model
other_model = timm.create_model('vit_base_patch16_224', pretrained=True)
other_model.eval()
other_model.to(device)

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
            # output = model(data)
            # pred = output.argmax(dim=1)
            # mask = pred.eq(target.view_as(pred))
            # correct += annotations[mask,:].to(dtype=torch.int).sum(dim=0)
            # total += annotations.to(dtype=torch.int).sum(dim=0)

            other_output = other_model(data)
            other_pred = other_output.argmax(dim=1)
            other_mask = other_pred.eq(target.view_as(other_pred))
            other_correct += other_annotations[other_mask,:].to(dtype=torch.int).sum(dim=0)
            other_total += other_annotations.to(dtype=torch.int).sum(dim=0)
finally:
    # # Compute accuracies per factor for resnet
    # factor_accs = (correct/total).cpu().detach().numpy()
    # results = pd.DataFrame({'Factor': FACTORS, 'acc': factor_accs}).sort_values('acc', ascending=False)

    # # Compute error ratios per factor
    # results['Error ratio'] = (1 - results['acc']) / (1-(correct.sum()/total.sum()).item())

    # # Plot results
    # # plots.plot_bar_plot(results, x='Factor', y='Error ratio')
    # # plt.show()
    # print('===RESNET==')
    # print(results)

    # Compute accuracies per factor for other model
    other_factor_accs = (other_correct/other_total).cpu().detach().numpy()
    other_results = pd.DataFrame({'Factor': FACTORS, 'acc': other_factor_accs}).sort_values('acc', ascending=False)

    # Compute error ratios per factor for other model
    other_results['Error ratio'] = (1 - other_results['acc']) / (1-(other_correct.sum()/other_total.sum()).item())

    # Plot results
    # plots.plot_bar_plot(other_results, x='Factor', y='Error ratio')
    # plt.show()
    print('===TIMM===')
    print(other_results)
    other_results.to_csv('output/imagenet_x_timm.csv')