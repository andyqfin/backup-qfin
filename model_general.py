import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import math
import subprocess

import torch.jit as jit
from sympy import sequence

from torchinfo import summary
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Parameter

from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

def check_cuda():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA available")
    else:
        print("CUDA not available")
        device = torch.device("cpu")

    return device

def general_setting_train(data, model):

    max_epoch = 300
    lr = 1e-3

    batch_size = 32

    criterion = nn.MSELoss()

    print('learning rate:', lr)

    torch.backends.cudnn.benchmark = True
    print(torch.backends.cudnn.version())

    nums_time, nums_dim = np.shape(data)[1:]

    summary(model, input_size=(batch_size, nums_time - 1, nums_dim ), depth=1)

    patience = 20
    best_loss = math.inf
    loss_save = np.array([])
    counter = 0

    return batch_size, lr, max_epoch, patience, criterion, best_loss, loss_save, counter


def save_checkpoint(loss_save, model, scheduler, model_file, save_loss, t_epoch_loss, c_epoch_loss, best_loss, counter):

    model_file3 = model_file + '_scheduler' + '.npy'
    model_file2 = model_file + '.npy'
    model_file = model_file + '.pth'

    np.save(model_file2, loss_save)

    save_loss = np.append(save_loss, [[t_epoch_loss], [c_epoch_loss]], axis=1)

    if c_epoch_loss < best_loss:

        torch.save(scheduler.state_dict(), model_file3)
        torch.save(model.state_dict(), model_file)
        best_loss = c_epoch_loss
        saved = True
        counter = 0
    else:
        counter = counter + 1
        saved = False

    return best_loss, saved, counter

def compute_loss(model, dataloader, criterion):

    model.eval()  # Set mod el to evaluation mode
    loss_ = 0

    nums_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            nums_samples = nums_samples + inputs.shape[0]

            outputs = model(inputs)
            loss = criterion(outputs, targets) * inputs.shape[0]
            loss_ += loss.item()

    avg_loss = loss_ / nums_samples

    return avg_loss

def learning_param(model, lr):

    optimizer = optim.Adam(model.parameters(), lr=lr)

    current_lr = lr
    scheduler = ReduceLROnPlateau(optimizer,mode='min', factor=0.2, patience=5,
                                                           threshold=0, threshold_mode='rel', eps=1e-8, cooldown=0, min_lr=0)

    return current_lr, optimizer, scheduler


def load_dataset_only(dataset, batch_size):

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_dataloader



# def create_random_split_loaders(dataset, batch_size):
#
#     indices = list(range(len(dataset)))
#     random.shuffle(indices)
#
#     train_len = int(0.8 * len(dataset))
#
#     train_subset = Subset(dataset, indices[:train_len])
#     val_subset = Subset(dataset, indices[train_len:])
#
#     train_loader_ = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#     val_loader_ = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
#
#
#     return train_loader_, val_loader_












#
# def train_model_lstm(model, data, model_file):
#
#     device = check_cuda()
#     batch_size, lr, max_epoch, patience, criterion, best_loss, loss_save, counter = general_setting_train(data, model)
#     current_lr, optimizer, scheduler = learning_param(model, lr)
#
#     c_epoch_loss = 0
#     save_loss = np.empty((2,0) )
#
#     for epoch in range(max_epoch):
#
#         train_dataloader, valid_dataloader = train_valid(data, batch_size, device)
#         loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)
#         model.train()
#         last_index = len(loop) - 1
#
#         for index, (inputs, targets) in loop:
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#
#             loop.set_description( f'Epoch[{epoch}]' )
#             loop.set_postfix(loss=loss.item())
#
#             if index == last_index:
#
#                 t_epoch_loss = compute_loss(model, train_dataloader, criterion)
#                 c_epoch_loss = compute_loss(model, valid_dataloader, criterion)
#
#                 best_loss, saved, counter = save_checkpoint(loss_save, model, scheduler, model_file, save_loss, t_epoch_loss, c_epoch_loss, best_loss, counter)
#
#                 current_lr = scheduler.get_last_lr()[0]
#                 scheduler.step(c_epoch_loss)
#                 bad = scheduler.state_dict()['num_bad_epochs']
#
#                 loop.set_postfix(loss=c_epoch_loss)
#                 loop.set_description(f'Epoch[{epoch}] LR: {current_lr:.1e} bad:{bad} train loss: {t_epoch_loss:.6e} valid loss: {c_epoch_loss:.6e} saved: {saved} patience: {counter} ')
#
#             optimizer.step()
#
#         if counter > patience:
#             print(f"Stopping early at epoch {epoch}")
#             break

    # print("Training complete.")
    #
    # return c_epoch_loss
