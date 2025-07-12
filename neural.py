import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimizations

import numpy as np
import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Loss

from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model


def objective_function(x, data):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

    return x1*x2**x3 - x2 ** x3 *x1* 99

def generalized_dtw_loss_function(x, data):

    dtw_x = 0

    rmse_dtw_loss = np.sqrt( np.mean( np.sum( (dtw_x) ** 2 ) ) )

# 自定義損失函數，接受額外參數
def custom_loss(data_param):

    def loss(y_true, y_pred):

        y_pred_reshaped = tf.reshape(y_pred, (y_pred.shape[0], 252, 1))

        # 將 data_param 轉換為 Tensor
        tensor_data = tf.convert_to_tensor(data_param, dtype=tf.float32)

        # 替換 compute_ndtw_loss 為平方函數
        # 這裡假設對 y_pred_reshaped 的每個元素取平方

        # 計算損失值（例如取平方後的平均值）
        loss_value = tf.sqrt(tf.reduce_mean( (y_pred_reshaped) ** 2) )

        # 打印損失值（可選）
        print(loss_value)

        return loss_value

    return loss

def neural_model():

    aa = 252 * 1

    model = Sequential()
    model.add(Input(shape=(aa,)))  # 使用 Input 物件作為第一層
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(aa, activation='relu'))

    return model

def call_optimzer(clustered_data):

    # print( np.shape( clustered_data ))
    
    combined = multi_matrix(clustered_data)

    model = neural_model()
    model.compile(optimizer='adam', loss=custom_loss(clustered_data), run_eagerly=True)

    # 定義 Early Stopping
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Encoder
    encoder_input = Input(shape=(1, 1))
    encoder_lstm = LSTM(64, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_input)
    
    # Decoder（使用 RepeatVector 擴展）
    decoder_input = RepeatVector(21)(state_h)
    decoder_lstm = LSTM(64, return_sequences=True)
    decoder_output = decoder_lstm(decoder_input, initial_state=[state_h, state_c])
    
    # 輸出序列
    output = TimeDistributed(Dense(1))(decoder_output)
    model = Model(encoder_input, output)


def neural_optimizer(current_data, clusters):

    print("perform neural optimizer")

    counter = 0

    for key, values in clusters.items():
        indices_str = " ".join(map(str, values))

        if ( key != -1 ):

            data = current_data[values, :, :]
            
            # print(key, " : ", values)
            
            call_optimzer(data)
            
        counter = counter + 1
        
    print( "numbers of labels : ", counter )
    
    print("===")
    
    return 0
