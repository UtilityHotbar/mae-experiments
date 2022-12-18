import random
import os

IMAGE_DIR = 'imagenet_val_folders'
TRAIN_DIR = 'imagenet/train'
VAL_DIR = 'imagenet/val'

with os.scandir(IMAGE_DIR) as it:
    for thing in it:
        if thing.is_dir():
            group_name = thing.name
            os.system(f'mkdir -p ./{TRAIN_DIR}/{group_name}')
            os.system(f'mkdir -p ./{VAL_DIR}/{group_name}')
            with os.scandir(IMAGE_DIR+'/'+group_name) as photos:
                cands = []
                for photo in photos:
                    if photo.is_file() and photo.name.endswith('.JPEG'):
                        cands.append(photo.name)
                random.shuffle(cands)
                train_val_split = int((len(cands)/10)*9)
                print(group_name, len(cands), train_val_split)
                train = cands[:train_val_split]
                val = cands[train_val_split:]
                for flnm in train:
                    os.system(f'cp ./{IMAGE_DIR}/{group_name}/{flnm} ./{TRAIN_DIR}/{group_name}')
                for flnm in val:
                    os.system(f'cp ./{IMAGE_DIR}/{group_name}/{flnm} ./{VAL_DIR}/{group_name}')