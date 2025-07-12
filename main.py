# Environment configuration (must be set before TF import)
import os
import sys

from sklearn.cluster import AgglomerativeClustering, HDBSCAN, MeanShift, Birch, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

from autoencoder_lstm import autoencoder_latent
from clustering import get_cluster, clustering_score, latent_dbscan, elbow_method, run_dbscan2, get_xmeans, \
    compare_kmeans_time
from giants import *
from dtw import *
from lstm_forecast import run_lstm_forecast_model

# from neural import neural_optimizer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TF info logs

# Core scientific computing
import numpy as np
import pandas as pd

import pickle
# Visualization setup
# import matplotlib.pyplot as plt

# set numpy display precision
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

import torch
from data import *
from fast_pytorch_kmeans import KMeans
from joblib import Parallel, delayed

def run_dtw(sequence, batch_size):

    gp = giants(GiantsParams(sequence, batch_size = batch_size ))
    
    return gp
    
    # clustering_score(gp512.labels, gp1024.labels, 'ari')
    # clustering_score(gp1024.labels, gp1024.labels, 'ari')


def run_autoencoder(sequence):

    print(np.shape(sequence)[1])

    layers = 1
    latent_dim = 128
    units = 1024
    # epoches_loss = autoencoder_latent('train', sequence, latent_dim, units, layers)
    print(np.shape(sequence))
    epoches_loss = autoencoder_latent('train', sequence, units, layers)

    return 0

def check_match(i, a, b):
    return np.any(np.all(a == b[i], axis=(1, 2)))

def main():

    print("main program starting")

    # marked total for start day and end time
    start_date = "2013-01-01"
    start_date = "1998-01-01"
    end_date = "2022-12-31"

    # original data
    org_data = load_data("aapl1998", start_date, end_date)

    # org_data = replace_zero_to_epsilon(org_data)

    next_day = start_date

    start_date, split_date, end_date = time_split_date(start_date, end_date, 0.95)

    lookback_days = 3

    lookback_date = lookback(split_date, lookback_days, 'minus')
    slide_date = lookback(start_date, lookback_days, 'add')

    print(slide_date)

    # train1, train2, train3, test1, test2, next_day= loop_date(next_day)
    volume_setting = 'no_volume'

    seq = seq_data(org_data, start_date, split_date, slide_date, lookback_days, setting=volume_setting)

    # seq2 = seq_data(org_data, lookback_date, end_date, lookback_days, setting=volume_setting)

    print('completed')

    # run_lstm_forecast_model('train', seq, seq2, 128, 1, 'tanh', volume_setting)

    # run_lstm_forecast_model('train', seq, seq2, 128, 3, 'tanh', volume_setting)


    # print(np.shape(sequence))
    # run_lstm_forecast_model('train', sequence, 1024, 3)

    # sequence = seq_data(org_data, train1, train3, 40, setting='no_volume')
    # print(np.shape(sequence))
    # run_autoencoder(sequence)
    #



    # result = run_autoencoder(sequence)
    
    # gp = run_dtw(sequence, 1024)

    # clustering_score(result.labels_, gp.labels, 'ari')
    
if __name__ == '__main__':

    main()



