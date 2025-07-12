from tensorflow.keras import layers, models, losses
import numpy as np

# 生成模擬數據（假設時間步長T=10，N=3）
timesteps = 10
n_features = 3
x = np.random.randn(100000, timesteps, n_features)  # 輸入數據
y = np.random.randn(100000, timesteps, 1)           # 目標輸出


# 編碼器部分
encoder_input = layers.Input(shape=(timesteps, n_features))
encoder_lstm1 = layers.LSTM(32, return_sequences=True)  # 只返回序列
encoder_output1 = encoder_lstm1(encoder_input)
encoder_lstm2 = layers.LSTM(32, return_sequences=True, return_state=True)
encoder_output, state_h, state_c = encoder_lstm2(encoder_output1)

# 主解碼器（Decoder1）
decoder_input = layers.RepeatVector(timesteps)(state_h)  # 複製最後的隱藏狀態
decoder_lstm1 = layers.LSTM(32, return_sequences=True)
decoder_output1 = decoder_lstm1(decoder_input, initial_state=[state_h, state_c])
output_main = layers.TimeDistributed(layers.Dense(1))(decoder_output1)

# 輔助解碼器（Decoder2，使用 Dense）
decoder2_input = layers.Concatenate(axis=-1)([decoder_output1])
decoder_dense = layers.Dense(32, activation="relu")(decoder2_input)  # 用 Dense 取代 LSTM
output_aux = layers.Dense(1)(decoder_dense)  # 產出最終輸出

# 完整模型
model = models.Model(
    inputs=encoder_input,
    outputs=[output_main, output_aux]
)

model.compile(optimizer='adam', loss='mse')
model.summary()

# 訓練模型
model.fit(x, [y, y], epochs=10, batch_size=32)
