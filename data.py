# Environment configuration (must be set before TF import)
import math
import os

from sklearn.preprocessing import MinMaxScaler

from technical_indicator import load_technical_indicators

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TF info logs

# Core scientific computing
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta, datetime
from numpy.lib.stride_tricks import sliding_window_view

def load_data(raw, startdate, enddate):

    subfolder = "data"

    txt = os.path.join(subfolder, raw + ".txt")
    pkl = os.path.join(subfolder, raw + ".pkl")

    stock_list = [line.strip() for line in open(txt, "r")]

    try:
        data = pd.read_pickle(pkl)

    except FileNotFoundError:
        data = yf.download(stock_list, start=startdate, end=enddate, interval="1d", auto_adjust=False, prepost=False, ignore_tz= False)
        data.to_pickle(pkl)

    return data

def loop_date(temp_start):

    train1 = pd.to_datetime(temp_start)
    train3 = train1 + pd.offsets.MonthEnd(12 * 10)

    test1 = train3 + pd.DateOffset(days=1)
    test2 = train3 + pd.offsets.MonthEnd(3)

    train2 = train3 - pd.offsets.MonthBegin(3)
    next_day = train1 + pd.offsets.MonthBegin(3)

    train1 = train1.strftime('%Y-%m-%d')
    train2 = train2.strftime('%Y-%m-%d')
    train3 = train3.strftime('%Y-%m-%d')
    test1 = test1.strftime('%Y-%m-%d')
    test2 = test2.strftime('%Y-%m-%d')

    print("training period: " + str(train1) + " , " + str(train2) + " , " + str(train3))
    print("testing period : " + str(test1) + " , " + str(test2) )

    return train1, train2, train3, test1, test2, next_day

def time_split_date(start_date, end_date, ratio):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    day_diff = (end_date - start_date).days

    split_date = start_date + timedelta(days=int(day_diff * ratio))

    start_date = start_date.strftime('%Y-%m-%d')
    split_date = split_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    print('start : ' + start_date + ' , ' + split_date + ' , ' + end_date)

    return start_date, split_date, end_date


def lookback(date, lookback, date_direction):

    date = datetime.strptime(date, '%Y-%m-%d')

    if date_direction == 'minus':
        date = date - timedelta(days=lookback) + timedelta(days=1)

    if date_direction == 'add':
        date = date + timedelta(days=lookback) - timedelta(days=1)

    date = date.strftime('%Y-%m-%d')

    print('lookback_date : ' + date)

    return date


def replace_zero_to_epsilon(data):

    # epsilon = 1e-10
    epsilon = 1
    data = data.replace(0, epsilon)
    
    data = np.log(data)

    return data

def get_pd_stock(data):

    adj = data['Adj Close']
    close = data['Close']
    high = data['High']
    low = data['Low']
    open = data['Open']
    volume = data['Volume']

    return adj, close, high, low, open, volume

def relative_price(data):

    adj, close, high, low, open, volume = get_pd_stock(data)

    s0, s1, s2, s3 = computes_relative_price(adj, close, high, low, open )
    s4, s5, s6, s7 = computes_relative_price(close, adj, high, low, open )
    s8, s9, s10, s11 = computes_relative_price(high, adj, close, low, open )
    s12, s13, s14, s15 = computes_relative_price(low, adj, close, high, open )
    s16, s17, s18, s19 = computes_relative_price(open, low, adj, close, high)

    data_value = (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19)

    data_value = np.concatenate(data_value, axis=2)

    return data_value

def compute_min_max(data):

    data = ( data - np.nanmin(data) ) / (np.nanmax(data) - np.nanmin(data) )

    return data

def computes_relative_price(data, *other):

    s0 = data_swap( np.log(data/other[0]) )
    s1 = data_swap( np.log(data/other[1]) )
    s2 = data_swap( np.log(data/other[2]) )
    s3 = data_swap( np.log(data/other[3]) )

    return s0, s1, s2, s3

def shift_first(x):

    x = x[1: , :]
    print(np.shape(x))
    return x


def swap_data(data, setting):

    adj, close, high, low, open, volume = get_pd_stock(data)
    reshaped_data = []

    adj = data_swap(adj)
    close = data_swap(close)
    high = data_swap(high)
    low = data_swap(low)
    open = data_swap(open)
    volume = data_swap(volume)

    return adj, close, high, low, open, volume

def compute_log_ratio(data):

    value = np.log(data[:, 1:, :]  / data[:, :-1, :] )

    return value


def compute_diff(data):

    value = data[:, 1:, :] - data[:, :-1, :]

    return value

def get_volume(data):

    value = np.array(data)[1:, :]
    log_value = np.log(value)

    return value, log_value

def func_swap_data(data, setting, func_set):

    adj, close, high, low, open, volume = swap_data(data, setting)

    func = 0

    if func_set == 'compute_diff':
        func = compute_diff

    adj = func(adj)
    close = func(close)
    high = func(high)
    low = func(low)
    open = func(open)
    volume = func(volume)

    con_data = np.concatenate((adj, close, high, low, open, volume), axis=2)

    return con_data

def data_log(data, train1, train2, slide_date, setting):

    mask = (data.index >= train1) & (data.index <= train2)
    data = data[mask]

    relative_3d = relative_price(data)

    adj, close, high, low, open, volume = swap_data(data, setting)

    data_diff = func_swap_data(data, setting, 'compute_diff')
    data_diff = func_swap_data(data, setting, 'compute_log_ratio')

    # load_technical_indicators(log_data, diff_data, log_data_v, diff_data_v)

    reshaped_data = np.concatenate((relative_3d, relative_3d), axis=2)

    return reshaped_data

def data_swap(data):
    #only for one dim each

    data = np.array(data)
    s0, s1 = np.shape(data)
    nums_stocks = s1 // 1
    nums_stocks = int(nums_stocks)
    data = data.reshape(s0, nums_stocks, 1).swapaxes(0, 1)

    return data


def gen_time_series_data(meta_data, time_step):

    print("generating time series data")

    print(np.shape(meta_data))

    num_samples = np.shape(meta_data)[0]  # Number of samples
    num_steps = np.shape(meta_data)[1] - time_step + 1  # Number of steps
    saved = np.zeros((num_samples * num_steps, time_step, np.shape(meta_data)[2]))

    for i in range(num_steps):
        x = meta_data[:, i:i + time_step, :]
        saved[i * num_samples:(i + 1) * num_samples, :, :] = x

    return saved

def seq_data(org_data, date, date2, slide_date, length, setting):

    data = data_log(org_data, date, date2, slide_date, setting=setting)

    seq_meta = gen_time_series_data(data, time_step = length)

    seq_meta = compute_min_max(seq_meta)

    non_nan_batches = ~np.any(np.isnan(seq_meta), axis=(1, 2))
    filtered_data = seq_meta[non_nan_batches]

    filtered_data = np.array(filtered_data)

    return filtered_data