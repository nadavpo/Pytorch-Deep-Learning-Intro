import os
import numpy as np
import copy
from tqdm import tqdm
import shutil
import random
from PIL import Image
import time

from models import get_model, save_model
from visualization import plot_train_results

import torch
import torch.utils.data
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

REORGANIZED_DATA = False
IMG_SIZE = 100
DATA_ROOT_DIR = "PetImages"
VAL_DIR_NAME = os.path.join(DATA_ROOT_DIR, "val")
TRAIN_DIR_NAME = os.path.join(DATA_ROOT_DIR, "train")
CLASS_NAMES = ["dog", "cat"]
NUM_TRAIN_EPOCHS = 20
STOP_IF_NOT_IMPROVE = 0
LOAD_MODEL = False
LOAD_MODEL_PATH = 'models/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DataOrganization:
    VAL_PERCENTS = 0.2

    def organized_data(self):
        data_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMG_SIZE, IMG_SIZE))
        ])

        for folder in [VAL_DIR_NAME, TRAIN_DIR_NAME]:
            if os.path.isdir(folder):
                shutil.rmtree(folder)
            os.mkdir(folder)
            [os.mkdir(os.path.join(folder, class_name)) for class_name in CLASS_NAMES]

        for class_name in CLASS_NAMES:
            class_files_names = []
            class_name_dir = os.path.join(DATA_ROOT_DIR, class_name)
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
            [shutil.copyfile(os.path.join(class_name_dir, f), os.path.join(VAL_DIR_NAME, class_name, f)) for f in
             tqdm(class_files_names[:n_images_for_val])]
            [shutil.copyfile(os.path.join(class_name_dir, f), os.path.join(TRAIN_DIR_NAME, class_name, f)) for f in
             tqdm(class_files_names[n_images_for_val:])]

            time.sleep(0.1)
            print(f"transform {class_name} data...")
            time.sleep(0.1)
            for phase_dir in [VAL_DIR_NAME, TRAIN_DIR_NAME]:
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


def create_dataloders():
    data_transforms = {'train': transforms.Compose([
        transforms.Grayscale(),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(45),
        transforms.ToTensor(),
    ]),
        'val': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])}

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_ROOT_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    return dataloaders


def run_epoch(model, mode, dataloader, loss_metric, optimizer=None):
    train = False
    if mode == 'train':
        model.train()
        train = True
    elif mode in ['val', 'validation', 'test']:
        model.eval()
    else:
        print("Unknown mode")
        return
    correct_class = 0
    total_loss = 0

    for _, (X, y) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            batch_loss = loss_metric(outputs, y)
            total_loss += batch_loss.item() * len(y)
            if train:
                batch_loss.backward()
                optimizer.step()
            correct_class += torch.sum(preds == y.data)
    return model, correct_class.double() / len(dataloader.dataset), total_loss / len(dataloader.dataset)


def train_and_eval(model, data_loader, num_epochs):
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # , weight_decay=1e-5)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    best_acc = 0
    best_loss = 0
    not_improve_epochs = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    losses_train = []
    losses_val = []
    accs_train = []
    accs_val = []
    for i in range(num_epochs):
        start_time = time.time()
        model, acc_train, loss_train = run_epoch(model, 'train', data_loader['train'], loss_function, optimizer)
        _, acc_val, loss_val = run_epoch(model, 'val', data_loader['val'], loss_function, optimizer)
        end_time = time.time()
        time.sleep(0.1)
        print(f"epoch {i} finished. Total time {round(end_time - start_time, 3)} sec")
        print(f"acc_train {round(acc_train.item(), 3)} loss_train {round(loss_train, 3)}"
              f" acc_val {round(acc_val.item(), 3)} loss_val {round(loss_val, 3)}")
        accs_train.append(acc_train)
        losses_train.append(loss_train)
        accs_val.append(acc_val)
        losses_val.append(loss_val)
        if acc_val > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = acc_val
            best_loss = loss_val
            not_improve_epochs = 0
        else:
            not_improve_epochs += 1
            if not_improve_epochs >= STOP_IF_NOT_IMPROVE > 0:
                break
        exp_lr_scheduler.step()
    model.load_state_dict(best_model_wts)
    return model, best_acc, best_loss, accs_train, losses_train, accs_val, losses_val


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    if REORGANIZED_DATA:
        d = DataOrganization()
        d.organized_data()

    data_loaders = create_dataloders()
    net, epoch = get_model(LOAD_MODEL, LOAD_MODEL_PATH, IMG_SIZE, len(CLASS_NAMES))

    net, best_acc, best_loss, accs_train, losses_train, accs_val, losses_val = \
        train_and_eval(net, data_loaders, NUM_TRAIN_EPOCHS)

    save_model(net, best_acc, epoch + len(accs_val))

    visualization(best_acc, best_loss, accs_train, losses_train, accs_val, losses_val)
