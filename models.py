import os
import time

import torch
import torch.utils.data
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TwoBlocksNet(nn.Module):
    def __init__(self, im_size, n_outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 128, 5)
        self.bn2 = nn.BatchNorm2d(128)

        #  this is for checking the automatically the input size for the first fc layer
        x = torch.randn(im_size, im_size).view(-1, 1, im_size, im_size)
        self.fc1_size = 0
        self.convs(x)
        ##
        self.fc1 = nn.Linear(self.fc1_size, 512)
        self.fc2 = nn.Linear(512, n_outputs)

    def convs(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(self.bn1(F.leaky_relu(self.conv2(x))), 2)

        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool2d(self.bn2(F.leaky_relu(self.conv4(x))), 2)

        if self.fc1_size == 0:
            self.fc1_size = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.fc1_size)
        x = F.leaky_relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


def get_model(load_model, path, im_size, n_outputs):
    # we can choose any model to send back from here
    net = TwoBlocksNet(im_size, n_outputs).to(device)
    epoch = 0
    # load pre-trained model and number of epoch that this model run on if ask to
    if load_model:
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            net.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
        else:
            print("model not found. Start training on new model")
    return net, epoch


def save_model(net, acc, epoch):
    # save
    torch.save({
        'epoch': epoch,
        'acc': acc,
        'model_state_dict': net.state_dict(),
    }, f'models/model_{time.time()}.pkl')

    # if accuracy is the better from that of the 'best_model', save tha current instead
    if not os.path.isfile('models/best_model/model.pkl'):
        if not os.path.isdir('models/best_model'):
            os.mkdir('models/best_model')
        torch.save({
            'epoch': epoch,
            'acc': acc,
            'model_state_dict': net.state_dict(),
        }, 'models/best_model/model.pkl')
    else: # if there is no 'best_model', save this as the best
        checkpoint = torch.load('models/best_model/model.pkl')
        bst_model_acc = checkpoint['acc']
        if bst_model_acc < acc:
            os.remove('models/best_model/model.pkl')
            torch.save({
                'epoch': epoch,
                'acc': acc,
                'model_state_dict': net.state_dict(),
            }, f'models/best_model/model.pkl')