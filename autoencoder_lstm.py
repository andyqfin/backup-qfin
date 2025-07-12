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

from model_general import check_cuda, general_setting_train, save_checkpoint, compute_loss, learning_param


class AutoencoderLayers(nn.Module):

    def __init__(self, input_dim, latent_dim, units, layers):
        super(AutoencoderLayers, self).__init__()
        self.encoder = nn.LSTM(input_dim, units, batch_first=True)
        self.encoder1 = nn.LSTM(units, units, num_layers=layers, batch_first=True)
        self.encoder_latent = nn.LSTM(units, latent_dim, batch_first=True)
        self.decoder_latent = nn.LSTM(latent_dim, units, num_layers=layers, batch_first=True)
        self.decoder1 = nn.LSTM(units, units, num_layers=layers, batch_first=True)
        self.decoder = nn.LSTM(units, input_dim, batch_first=True)

    def forward(self, x):

        x, _ = self.encoder(x)
        x, _ = self.encoder1(x)

        _, (prev_hidden, _) = self.encoder_latent(x)
        batch_size, dim, _ = x.shape
        repeated_hidden = prev_hidden.repeat_interleave(dim, dim=1)
        split_tensors = repeated_hidden.view(batch_size, -1, repeated_hidden.shape[-1])
        x, _ = self.decoder_latent(split_tensors)

        x, _ = self.decoder1(x)
        x, _ = self.decoder(x)
        return x

def autoencoder_train_valid(data, batch_size, device):

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    dataset = TensorDataset(data_tensor, data_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_dataloader, val_dataloader

def autoencoder_train_only(data, batch_size, device, shuffle):

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    dataset = TensorDataset(data_tensor, data_tensor)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return train_dataloader

def train_autoencoder(model, data, model_file):

    device = check_cuda()
    batch_size, lr, max_epoch, patience, criterion, best_loss, loss_save, counter = general_setting_train(data, model)
    current_lr, optimizer, scheduler = learning_param(model, lr)

    c_epoch_loss = 0
    save_loss = np.empty((2,0) )

    for epoch in range(max_epoch):

        train_dataloader, valid_dataloader = autoencoder_train_valid(data, batch_size, device)
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)
        model.train()
        last_index = len(loop) - 1

        for index, (inputs, targets) in loop:

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            loop.set_description( f'Epoch[{epoch}]' )
            loop.set_postfix(loss=loss.item())

            if index == last_index:

                t_epoch_loss = compute_loss(model, train_dataloader, criterion)
                c_epoch_loss = compute_loss(model, valid_dataloader, criterion)

                best_loss, saved, counter = save_checkpoint(loss_save, model, scheduler, model_file, save_loss, t_epoch_loss, c_epoch_loss, best_loss, counter)

                current_lr = scheduler.get_last_lr()[0]
                scheduler.step(c_epoch_loss)
                bad = scheduler.state_dict()['num_bad_epochs']

                loop.set_postfix(loss=c_epoch_loss)
                loop.set_description(f'Epoch[{epoch}] LR: {current_lr:.1e} bad:{bad} train loss: {t_epoch_loss:.6e} valid loss: {c_epoch_loss:.6e} saved: {saved} patience: {counter} ')

            optimizer.step()

        if counter > patience:
            print(f"Stopping early at epoch {epoch}")
            break

    print("Training complete.")

    return c_epoch_loss

def autoencoder_latent(setting, data, latent_dim, units, layers):

    timesteps, input_dim = data.shape[1], data.shape[2]
    model_name = f'AE(LSTM)_{timesteps}_{latent_dim}_{units}_{layers}'

    model = AutoencoderLayers(input_dim, latent_dim, units, layers)

    if setting == 'train':
        c_epoch_loss = train_autoencoder(model, data, model_name)
        return c_epoch_loss
    if setting == 'load':
        latent = load_autoencoder(model, model_name, data)
        return latent
    if setting == 'restore':
        a = 0

    return 0

def load_autoencoder(model, model_file, data):

    device = check_cuda()
    batch_size, lr, max_epoch, patience, criterion, best_loss, loss_save, counter = general_setting_train(data, model)

    model_name = model_file + '.pth'
    print(model_name)

    model.load_state_dict(torch.load(model_name))
    model.to(device)
    model.eval()

    dataloader = autoencoder_train_only(data, batch_size, check_cuda(), shuffle = False)
    latent_list = []

    loss = compute_loss(model, dataloader, criterion)
    print(loss)

    t_epoch_loss = compute_loss(model, dataloader, criterion)

    with torch.no_grad():
        for index, (inputs, targets) in enumerate(dataloader):
            latent = encoder_only(model, inputs)
            latent_list.append(latent)

    latent_list = np.vstack(latent_list)
    latent_list = np.array(latent_list)

    return latent_list

def encoder_only(model, batch):

    x, _ = model.encoder(batch)
    x, _ = model.encoder1(x)
    _, (prev_hidden, _) = model.encoder_latent(x)

    latent_space = prev_hidden.squeeze(0).cpu().numpy()

    return latent_space

def load_checkpoint(model, scheduler):

    model_state = torch.load('model.npy')
    model.load_state_dict(model_state)

    # Load scheduler state
    scheduler_state = torch.load('scheduler.npy')
    scheduler.load_state_dict(scheduler_state)
