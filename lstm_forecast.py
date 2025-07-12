import random

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

from model_general import check_cuda, general_setting_train, compute_loss, learning_param, \
    save_checkpoint, load_dataset_only

import torch
from torch.utils.data import Dataset

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
import torch.distributed as dist

import argparse

import torch.distributed as dist
class ReLULSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(ReLULSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.w_ih = nn.ModuleList()
        self.w_hh = nn.ModuleList()

        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.w_ih.append(nn.Linear(input_dim, 4 * hidden_size))
            self.w_hh.append(nn.Linear(hidden_size, 4 * hidden_size))

    def forward(self, input, hidden=None):

        batch_size, seq_len, _ = input.size()

        if hidden is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=input.device) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=input.device) for _ in range(self.num_layers)]
        else:
            h, c = list(hidden[0]), list(hidden[1])

        outputs = []

        for t in range(seq_len):
            x = input[:, t, :]
            hy, cy = [], []
            for i in range(self.num_layers):
                gates = self.w_ih[i](x) + self.w_hh[i](h[i])
                i_gate, f_gate, c_gate, o_gate = gates.chunk(4, dim=1)

                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                c_gate = F.relu(c_gate)
                o_gate = torch.sigmoid(o_gate)

                c_new = f_gate * c[i] + i_gate * c_gate
                h_new = o_gate * torch.tanh(c_new)

                x = self.dropout(h_new)

                hy.append(h_new)
                cy.append(c_new)

            h, c = hy, cy
            outputs.append(h[-1].unsqueeze(1))  # 只保留最后一层输出

        output = torch.cat(outputs, dim=1)  # [batch, seq_len, hidden_size]
        h_n = torch.stack(h)  # [num_layers, batch, hidden_size]
        c_n = torch.stack(c)  # [num_layers, batch, hidden_size]

        return output, (h_n, c_n)

# Epoch[0] LR: 1.0e-03 bad:0 train loss: 9.086497e-05 valid loss: 3.694395e-04 saved: True patience: 0 : 100%|██████████| 5467/5467 [1:28:40<00:00,  1.03it/s, loss=0.000369]
# Epoch[1] LR: 1.0e-03 bad:1 train loss: 9.185410e-05 valid loss: 3.697643e-04 saved: False patience: 1 : 100%|██████████| 5467/5467 [1:28:48<00:00,  1.03it/s, loss=0.00037]
# Epoch[2] LR: 1.0e-03 bad:0 train loss: 9.014310e-05 valid loss: 3.685485e-04 saved: True patience: 0 : 100%|██████████| 5467/5467 [1:28:53<00:00,  1.02it/s, loss=0.000369]
# Epoch[3] LR: 1.0e-03 bad:1 train loss: 9.832145e-05 valid loss: 3.740833e-04 saved: False patience: 1 : 100%|██████████| 5467/5467 [1:28:51<00:00,  1.03it/s, loss=0.000374]


class TimeSeriesDataset(Dataset):
    def __init__(self, data, out_shift_nums):

        self.x = data[:, :-1, :]  # [N, T-1, D]
        self.y = data[:, -1:, 0:out_shift_nums]   # [N, D]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

from joblib import Parallel, delayed

class Forecast_model_lstm(nn.Module):
    def __init__(self, input_dim, units, layers, dropout, act, out_shift_nums):
        super(Forecast_model_lstm, self).__init__()
        print(input_dim)
        self.input = nn.LSTM(input_dim, units, batch_first=True)

        if act == 'relu':
            self.hidden = ReLULSTM(units, units, num_layers=layers, dropout=dropout)

        if act == 'tanh':
            self.hidden = nn.LSTM(units, units, num_layers=layers, dropout=dropout)

        self.linear = nn.Linear(units, out_shift_nums)

    def forward(self, x):

        x, _ = self.input(x)  # 先通过标准 LSTM 输入层
        x, _ = self.hidden(x)  # 然后通过自定义 ReLU LSTM 层

        x = x[:, -1, :].unsqueeze(1)
        x = self.linear(x)
        return x

def load_lstm_forecast(model, model_file, data):

    return 0

def run_lstm_forecast_model(setting, data, data2, units, layers, act, volume_setting):

    out_shift_nums =  -1

    if volume_setting == 'no_volume':
        out_shift_nums = 5
    if volume_setting == 'yes_volume':
        out_shift_nums = 6

    print(np.shape(data))

    timesteps, input_dim = data.shape[1], data.shape[2]
    model_name = f'Forecast_model_lstm_{timesteps}_{units}_{layers}_{act}'

    model = Forecast_model_lstm(input_dim, units, layers, 0, act, out_shift_nums)

    if setting == 'train':
        c_epoch_loss = train_lstm_forecast(model, data, data2, model_name, out_shift_nums)
        return c_epoch_loss
    if setting == 'load':
        latent = load_lstm_forecast(model, model_name, data)
        return latent
    if setting == 'restore':
        a = 0

    return 0

def train_lstm_forecast(model, data, data2, model_file, out_shift_nums):

    device = check_cuda()
    model = model.to(device)

    batch_size, lr, max_epoch, patience, criterion, best_loss, loss_save, counter = general_setting_train(data, model)
    current_lr, optimizer, scheduler = learning_param(model, lr)

    c_epoch_loss = 0
    save_loss = np.empty((2,0) )

    data = torch.tensor(data, dtype=torch.float32).to(device)
    data2 = torch.tensor(data2, dtype=torch.float32).to(device)

    dataset = TimeSeriesDataset(data, out_shift_nums)
    dataset2 = TimeSeriesDataset(data2, out_shift_nums)

    train_dataloader = load_dataset_only(dataset, batch_size)
    valid_dataloader = load_dataset_only(dataset2, batch_size)

    for epoch in range(max_epoch):

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
