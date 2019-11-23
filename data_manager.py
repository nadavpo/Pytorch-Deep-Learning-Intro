import os
import numpy as np
import copy
from tqdm import tqdm
import shutil
import random
from PIL import Image
import time

import torch
import torch.utils.data
import torch.multiprocessing
from torchvision import datasets, transforms


class DataOrganization:
    VAL_PERCENTS = 0.2

    # organized data for the Pytorch's data loaders
    def organized_data(self, img_size, val_dir_name, train_dir_name, class_names, data_root_dir):
        # transforms on the data before saving it
        data_transforms = get_transforms('pre save')

        for folder in [val_dir_name, train_dir_name]:
            if os.path.isdir(folder):
                shutil.rmtree(folder)
            os.mkdir(folder)
            [os.mkdir(os.path.join(folder, class_name)) for class_name in class_names]

        for class_name in class_names:
            class_files_names = []
            class_name_dir = os.path.join(data_root_dir, class_name)
            for f in os.listdir(class_name_dir):
                stat_info = os.stat(os.path.join(class_name_dir, f))
                if not (stat_info.st_size > 0):
                    print(f"{f} is not illegal image")
                    continue
                if '.jpg' in f:
                    try:
                        im = Image.open(os.path.join(class_name_dir, f))
                        class_files_names.append(f)
                    except:
                        print(f"{f} is illegal image")
            n_images = len(class_files_names)
            n_images_for_val = round(n_images * self.VAL_PERCENTS)
            random.shuffle(class_files_names)

            time.sleep(0.1)
            print(f"copy {class_name} data...")
            time.sleep(0.1)
            [shutil.copyfile(os.path.join(class_name_dir, f), os.path.join(val_dir_name, class_name, f)) for f in
             tqdm(class_files_names[:n_images_for_val])]
            [shutil.copyfile(os.path.join(class_name_dir, f), os.path.join(train_dir_name, class_name, f)) for f in
             tqdm(class_files_names[n_images_for_val:])]

            time.sleep(0.1)
            print(f"transform {class_name} data...")
            time.sleep(0.1)
            for phase_dir in [val_dir_name, train_dir_name]:
                root_dir = os.path.join(phase_dir, class_name)
                for f in tqdm(os.listdir(root_dir)):
                    if '.jpg' in f:
                        try:
                            im = Image.open(os.path.join(class_name_dir, f))
                            im = np.asarray(im)
                            im = Image.fromarray(np.uint8(im))
                            im = data_transforms(im)
                            im.save(os.path.join(root_dir, f))
                        except:
                            print(f"{f} is illegal image")


def create_dataloders(data_root_dir):
    data_transforms = {'train': get_transforms('train'),
                       'val': get_transforms('val')}

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_root_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    return dataloaders


def get_transforms(transform_type, img_size=100):
    transforms_dict = {'pre save':
        transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size))
        ]),
        'train':
            transforms.Compose([
                transforms.Grayscale(),
                # transforms.ColorJitter(brightness=0.5, contrast=0.5),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(45),
                transforms.ToTensor(),
            ]),
        'val':
            transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
        'test':
            transforms.Compose([
                transforms.ToTensor(),
            ])
    }

    return transforms_dict[transform_type]
