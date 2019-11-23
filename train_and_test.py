import torch
import torch.utils.data
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from tqdm import tqdm
import time
import os
import random
from data_manager import get_transforms
import matplotlib.pyplot as plt
from PIL import Image
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def train_and_eval(model, data_loader, num_epochs, stop_if_not_improve):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # , weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
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
            if not_improve_epochs >= stop_if_not_improve > 0:
                break
        exp_lr_scheduler.step()
    model.load_state_dict(best_model_wts)
    return model, best_acc, best_loss, accs_train, losses_train, accs_val, losses_val


def test(model, path_to_test, classes, n_images_to_test=0):
    transform = get_transforms('test')
    if os.path.isdir(path_to_test):
        if n_images_to_test <= 0:
            n_images_to_test = 5
        ims_to_check = random.sample([x for x in os.listdir(path_to_test) if "jpg" in x], k=n_images_to_test)
        if math.sqrt(n_images_to_test)%int(math.sqrt(n_images_to_test)) == 0:
            plots1 = int(math.sqrt(n_images_to_test))
            plots2 = n_images_to_test//plots1
        else:
            plots1 = int(math.sqrt(n_images_to_test))
            plots2 = n_images_to_test//plots1 + 1
        fig, axs = plt.subplots(plots1, plots2)
        fig.suptitle("Model's' results")
        for i in range(n_images_to_test):
            X = transform(Image.open(os.path.join(path_to_test, ims_to_check[i])))
            outputs = model(X.view(-1, X.shape[0], X.shape[1], X.shape[2]).to(device))
            _, preds = torch.max(outputs, 1)
            axs[i//plots2, i%plots2].imshow(X[-1, :, :])
            axs[i//plots2, i%plots2].set_title('model predict is - {0}'.format(get_key(preds, classes)))
        plt.show()

    elif os.path.isfile(path_to_test):
        fig, ax = plt.subplots()
        X = transform(Image.open(os.path.join(path_to_test, path_to_test)))
        outputs = model(X.view(-1, X.shape[0], X.shape[1], X.shape[2]).to(device))
        _, preds = torch.max(outputs, 1)
        ax.imshow(X[-1, :, :])
        ax.set_title('model predict is - {0}'.format(get_key(preds, classes)))
        plt.show()


def get_key(val, my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key
