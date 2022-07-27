import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd

# グラフ描画
from matplotlib import pylab as plt

# グラフを横長にする
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

def lstm(num_hidden=100, window_size=10, num_in_out=1):
    # ネットワークの構築
    model = Sequential() 
    model.add(LSTM(num_hidden, batch_input_shape=(None, window_size, num_in_out)))
    model.add(Dense(num_in_out)) 
    
    #コンパイル
    model.compile(loss='mean_squared_error', optimizer=Adam() , metrics = ['accuracy'])
    model.summary()
    return model

def learn(model, x_train, y_train, x_test, y_test, batch_size=20, n_epoch=150):
    hist = model.fit(x_train, y_train,
                        epochs=n_epoch,
                        validation_data=(x_test, y_test),
                        verbose=0,
                        batch_size=batch_size)
    # 損失値(Loss)の遷移のプロット
    plt.plot(hist.history['loss'],label="train set")
    plt.plot(hist.history['val_loss'],label="test set")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def future(model, x, pred_length, norm_scale, window_size=12, gridline_space=12):
    future_pred = x[:, 0, 0].copy()
    for tmp in range(pred_length):
        x_future_pred = future_pred[-1*window_size:]
        y_future_pred = model.predict(x_future_pred.reshape(1, window_size, 1))
        future_pewd = np.append(future_pred, y_future_pred)

    plt.plot(x[:,0,0] * norm_scale, color='blue',  label="observed")  # 実測値
    plt.plot(range(len(x),len(x)+pred_length), future_pred[-1*pred_length:] * norm_scale,  color='red',  label="feature pred")   # 予測値
    plt.legend()
    plt.xticks(np.arange(0, 145+pred_length, gridline_space)) # 12ヶ月ごとにグリッド線を表示
    plt.grid()
    plt.show()
