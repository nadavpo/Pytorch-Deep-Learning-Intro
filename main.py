import os
import torch.multiprocessing

from models import get_model, save_model
from visualization import plot_train_results
from data_manager import create_dataloders, DataOrganization
from train_and_test import train_and_eval, test

REORGANIZED_DATA = False
IMG_SIZE = 100
DATA_ROOT_DIR = "PetImages"
VAL_DIR_NAME = os.path.join(DATA_ROOT_DIR, "val")
TRAIN_DIR_NAME = os.path.join(DATA_ROOT_DIR, "train")
CLASS_NAMES = ["dog", "cat"]
NUM_TRAIN_EPOCHS = 10
STOP_IF_NOT_IMPROVE = 0
LOAD_MODEL = True
LOAD_MODEL_PATH = 'models/model_1574546050.9673572.pkl'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    if REORGANIZED_DATA:
        d = DataOrganization()
        d.organized_data(IMG_SIZE, VAL_DIR_NAME, TRAIN_DIR_NAME, CLASS_NAMES, DATA_ROOT_DIR)

    data_loaders = create_dataloders(DATA_ROOT_DIR)
    net, epoch = get_model(LOAD_MODEL, LOAD_MODEL_PATH, IMG_SIZE, len(CLASS_NAMES))

    net, best_acc, best_loss, accs_train, losses_train, accs_val, losses_val = \
        train_and_eval(net, data_loaders, NUM_TRAIN_EPOCHS, STOP_IF_NOT_IMPROVE)

    save_model(net, best_acc, epoch + len(accs_val))

   # plot_train_results(best_acc, best_loss, accs_train, losses_train, accs_val, losses_val)

    test(net, os.path.join(VAL_DIR_NAME, CLASS_NAMES[0]), data_loaders['train'].dataset.class_to_idx, 10)
    test(net, os.path.join(VAL_DIR_NAME, CLASS_NAMES[1]), data_loaders['train'].dataset.class_to_idx, 10)

