import pickle
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2
import matplotlib.pyplot as plt

INTENSITY = 0.5
DATA_PATH = 'heatmap_results/results.pickle'
BASE_DATA_PATH = 'heatmap_results/base_results.pickle'
IMG_PATH = 'heatmap_images/val'
PROCESSED_IMG_PATH = 'heatmap_images/processed'

with open(BASE_DATA_PATH, 'rb') as g:
    base_results = pickle.load(g)

with open(DATA_PATH, 'rb') as g:
    results = pickle.load(g)

print('Data loaded.')

# image_paths = ['heatmap_images/val/n01440764/ILSVRC2012_val_00000293.JPEG', 'heatmap_images/val/n01443537/ILSVRC2012_val_00000236.JPEG', 'heatmap_images/val/n01484850/ILSVRC2012_val_00002338.JPEG', 'heatmap_images/val/n01491361/ILSVRC2012_val_00002922.JPEG', 'heatmap_images/val/n01494475/ILSVRC2012_val_00001676.JPEG', 'heatmap_images/val/n01496331/ILSVRC2012_val_00000921.JPEG', 'heatmap_images/val/n01498041/ILSVRC2012_val_00001935.JPEG', 'heatmap_images/val/n01514668/ILSVRC2012_val_00000329.JPEG', 'heatmap_images/val/n01514859/ILSVRC2012_val_00001114.JPEG', 'heatmap_images/val/n01518878/ILSVRC2012_val_00001031.JPEG', 'heatmap_images/val/n01530575/ILSVRC2012_val_00000651.JPEG', 'heatmap_images/val/n01531178/ILSVRC2012_val_00000570.JPEG', 'heatmap_images/val/n01532829/ILSVRC2012_val_00000873.JPEG', 'heatmap_images/val/n01534433/ILSVRC2012_val_00000247.JPEG', 'heatmap_images/val/n01537544/ILSVRC2012_val_00000414.JPEG', 'heatmap_images/val/n01558993/ILSVRC2012_val_00001598.JPEG']
image_paths = []

for thing in os.scandir(IMG_PATH):
    if thing.is_dir():
        image_paths.append(IMG_PATH+'/'+thing.name+'/'+os.popen(f'ls {IMG_PATH}/{thing.name}').read().strip('\n'))

IM_RES_DIR = []

for path in image_paths:
    IM_RES_DIR.append({'path': path})

index_elem = 0
for elem in base_results[0]['raw']:
    IM_RES_DIR[index_elem]['base'] = elem
    index_elem += 1

for patch in results:
    r_dict = results[patch]['raw']
    index_elem = 0
    for elem in r_dict:
        IM_RES_DIR[index_elem][patch] = elem
        index_elem += 1

for image in IM_RES_DIR:
    base_prediction = np.argmax(image['base'])
    base_confidence = image['base'][base_prediction]
    conf_deltas = []
    for i in range(167):
        curr_pred = np.argmax(image[i])
        if curr_pred != base_prediction:
            print('PREDICTION SHIFT')
            conf_deltas.append(-1)
        else:
            # We presume the confidence has gone down
            conf_deltas.append(float(base_confidence-image[i][curr_pred]))
    image['deltas'] = torch.Tensor(conf_deltas+[0.0])

i = 0
for path in image_paths:

    # We process an image as if it were about to get tokenised to get the same aspect ratio
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    t = []
    crop_pct = 224 / 256
    size = int(224 / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(224))
    eval_transform = transforms.Compose(t)
    img = Image.open(path)
    eval_transform(img)
    img = np.array(img)

    # Turning confidence deltas into a heatmap
    heatmap = np.maximum(np.array(IM_RES_DIR[i]['deltas']), 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((14,12))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    img = heatmap * INTENSITY + img

    new_path = PROCESSED_IMG_PATH+'/'+'/'.join(path.split('/')[3:])
    cv2.imwrite(new_path, img)
    i += 1
