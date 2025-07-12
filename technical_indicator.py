import sys
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def check_nan(x):

    if np.isnan(x).any():
        return np.nan

    return x

def load_technical_indicators(log_data, diff_data, log_data_v, diff_data_v, vol, log_vol):

    print('load_technical_indicators')
    print(np.shape(log_data))

    sma = sma_func(log_data, 3)
    ema = ema_func(log_data, 3)

    print(np.shape(sma))
    # print(np.shape(ema))

    dif, dem, osc = macd_func(log_data, 12, 26, 9)

    # print(np.shape(dif))
    # print(np.shape(dem))
    # print(np.shape(osc))

    rsi = rsi_func(diff_data, 14)
    log_rsi = rsi_func(log_data, 14)

    # typical_price_func(x)

    return 0

def sma_func(x, period):

    slide = sliding_window_view(x, window_shape=period, axis=1).transpose(0, 1, 3, 2)

    sma = np.apply_along_axis(simple_moving_average, axis=2, arr= slide)

    sub_period = period - 1
    nan_pad = np.full(( np.shape(sma)[0], sub_period, np.shape(sma)[2]), np.nan)
    sma = np.concatenate([nan_pad, sma], axis=1)

    return sma

def simple_moving_average(x):

    if np.isnan(x).any():
        return np.nan

    return np.mean(x)

def ema_func(x, period):

    slide = sliding_window_view(x, window_shape=3, axis=1).transpose(0, 1, 3, 2)
    ema = np.apply_along_axis(exponential_moving_average, axis=2, arr=slide)

    sub_period = period - 1
    nan_pad = np.full(( np.shape(ema)[0], sub_period, np.shape(ema)[2]), np.nan)
    ema = np.concatenate([nan_pad, ema], axis=1)

    return ema

def exponential_moving_average(x):

    if np.isnan(x).any():
        return np.nan

    span = np.shape(x)[0]
    alpha = 2 / (span + 1)
    result = np.empty_like(x)
    result[0] = x[0]

    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]

    return result[-1]

def macd_func(x, short_period, long_period, signal_period):

    # moving_average_convergence_divergence standard setting
    # 12 periods, short period, fast period
    # 26 periods, long_period, slow period
    # dif = fast ema - slow ema
    # signal = 9, ema the dif => macd

    if short_period >= long_period:
        warnings.warn("macd, short_period should not >= long_period", UserWarning)
        sys.exit("program end")

    print('moving average convergence divergence')

    fast_slide = sliding_window_view(x, window_shape=short_period, axis=1).transpose(0, 1, 3, 2)
    long_ema = np.apply_along_axis(exponential_moving_average, axis=2, arr= fast_slide)

    slow_slide = sliding_window_view(x, window_shape=long_period, axis=1).transpose(0, 1, 3, 2)
    short_ema = np.apply_along_axis(exponential_moving_average, axis=2, arr= slow_slide)

    min_len = min(short_ema.shape[1], long_ema.shape[1])
    dif = short_ema[:, -min_len:, :] - long_ema[:, -min_len:, :]

    signal_slide = sliding_window_view(dif, window_shape=signal_period, axis=1).transpose(0, 1, 3, 2)
    dem = np.apply_along_axis(exponential_moving_average, axis=2, arr= signal_slide)

    dif_period = long_period - 1
    nan_pad_dif = np.full(( np.shape(x)[0], dif_period, np.shape(x)[2]), np.nan)
    arr_padded_dif = np.concatenate([nan_pad_dif, dif], axis=1)

    sub_period = long_period - 1 + signal_period - 1
    nan_pad_dem = np.full(( np.shape(x)[0], sub_period, np.shape(x)[2]), np.nan)

    arr_padded_dem = np.concatenate([nan_pad_dem, dem], axis=1)

    osc = arr_padded_dif - arr_padded_dem

    return arr_padded_dif, arr_padded_dem, osc

def rsi_func(x, period):

    print('rsi')
    slide = sliding_window_view(x, window_shape=period, axis=1).transpose(0, 1, 3, 2)

    gains = np.where(slide > 0, slide, 0)
    loss = np.where(slide < 0, -slide, 0)

    up_ema = np.apply_along_axis(exponential_moving_average, axis=2, arr= gains)
    down_ema = np.apply_along_axis(exponential_moving_average, axis=2, arr= loss)

    rs = up_ema / ( down_ema + 1e-300)
    rsi = 100 - (100 / (1 + rs) )

    sub_period = period - 1
    nan_pad = np.full(( np.shape(rsi)[0], sub_period, np.shape(rsi)[2]), np.nan)
    rsi = np.concatenate([nan_pad, rsi], axis=1)

    return rsi

def typical_price_func(x):

    print('typical price')

    tp = typical_price(x)

    print(tp)
    print(np.shape(tp))
    return 0

def typical_price(x):

    # if setting == 'no_volume':
    #     reshaped_data = np.concatenate((adj, close, high, low, open), axis=2)
    #
    # if setting == 'yes_volume':
    #     reshaped_data = np.concatenate((adj, close, high, low, open, volume), axis=2)

    # ( log low + log high + log close ) / 3
    # positive_money_flow = log volume * tp_up
    # negative_money_flow = log volume * tp_down
    # money_ratio  = positive_money_flow / negative_money_flow
    # mfi =  100 - (100 / (1 + money_ratio) )

    tp = ( x[:,:,0] + x[:,:,2] + x[:,:,3] ) / 3
    tp = tp[:, :, np.newaxis]

    return tp

# import numpy as np
# import matplotlib.pyplot as plt
#
# # 模擬 10 天的絕對價格與成交量資料
# high      = np.array([101, 103, 104, 105, 107, 106, 89, 109, 89, 113])
# low       = np.array([99, 100, 101, 102, 103, 102, 86, 105, 86, 109])
# close     = np.array([100, 102, 103, 104, 106, 105, 88, 108, 88, 112])
# volume    = np.array([1500, 1600, 1700, 1650, 1750, 1800, 1900, 1600, 2000, 2050])
#
# # ============== 傳統 MFI 計算 ====================
# typical_price = (high + low + close) / 3
# raw_money_flow = typical_price * volume
#
# tp_diff = typical_price[1:] - typical_price[:-1]
# positive_flow = np.where(tp_diff > 0, raw_money_flow[1:], 0)
# negative_flow = np.where(tp_diff < 0, raw_money_flow[1:], 0)
#
# money_flow_ratio = np.sum(positive_flow) / (np.sum(negative_flow) + 1e-6)
# mfi = 100 - 100 / (1 + money_flow_ratio)
#
# # ============== Log-MFI 計算 =====================
# log_high   = np.log(high)
# log_low    = np.log(low)
# log_close  = np.log(close)
# log_volume = np.log(volume)
#
# log_tp = (log_high + log_low + log_close) / 3
# log_tp_change = log_tp[1:] - log_tp[:-1]
# # log_volume = log_volume[1:]
# log_volume = log_volume[1:] - log_volume[:-1]
#
# log_volume = np.abs(log_volume)
#
# log_pos_flow = np.where(log_tp_change > 0,  log_tp_change, 0)
# log_neg_flow = np.where(log_tp_change < 0,  -log_tp_change, 0)
#
# log_pos_flow = log_pos_flow * log_volume
# log_neg_flow = log_neg_flow * log_volume
# log_money_ratio = np.sum(log_pos_flow) / (np.sum(log_neg_flow) + 1e-6)
# print(log_tp_change)
# print(log_volume)
# log_mfi = 100 - 100 / (1 + log_money_ratio)
#
# # ============== 結果比較 =====================
# print(f"傳統 MFI 指標：{mfi:.2f}")
# print(f"對數 Log-MFI 指標：{log_mfi:.2f}")
#
# # ============== 圖表呈現 =====================
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, 10), typical_price[1:], label='典型價格', color='gray', linestyle='--')
# plt.axhline(y=typical_price.mean(), color='lightgray', linestyle='dotted')
#
# plt.bar(0, mfi, width=0.4, label='傳統 MFI', color='blue')
# plt.bar(1, log_mfi, width=0.4, label='對數 Log-MFI', color='green')
#
# plt.xticks([0, 1], ['MFI', 'Log-MFI'])
# plt.title("傳統 MFI vs 對數 MFI")
# plt.ylabel("指標值 (0–100範圍)")
# plt.legend()
# plt.grid(True)
# plt.show()