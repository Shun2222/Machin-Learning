from .libs.lstm import *
from .libs.make_dataset import *
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    filepath = '../datas/AirPassengers.csv'
    data = pd.read_csv(filepath)
    data.head()

    # 型変換
    input_data = data['Passengers'].values.astype(float)
    print("input_data : " , input_data.shape ,type(input_data))

    # スケールの正規化
    norm_scale = input_data.max()
    input_data /= norm_scale

    window_size = 12
    x, y = make_dataset(input_data, window_size)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)    
    
    num_hidden = 50
    num_in_out = 1
    model = lstm(num_hidden, window_size, num_in_out)
    
    batch_size = 20
    n_epoch = 150
    learn(model, x_train, y_train, batch_size, n_epoch)

    # 予測
    y_pred_train = lstm_model.predict(X_train)
    y_pred_test = lstm_model.predict(X_test)
    
    # RMSEで評価
    # 参考：https://deepage.net/deep_learning/2016/09/17/tflearn_rnn.html
    def rmse(y_pred, y_true):
        return np.sqrt(((y_true - y_pred) ** 2).mean())
    print("RMSE Score")
    print("  train : " , rmse(y_pred_train, y_train))
    print("  test : " , rmse(y_pred_test, y_test))
    
    # 推定結果のプロット
    plt.plot(X[:, 0, 0], color='blue',  label="observed")  # 元データ
    plt.plot(y_pred_train, color='red',  label="train")   # 予測値（学習）
    plt.plot(range(len(X_train),len(X_test)+len(X_train)),y_pred_test, color='green',  label="test")   # 予測値（検証）
    plt.legend()
    plt.xticks(np.arange(0, 145, 12)) # 12ヶ月ごとにグリッド線を表示
    plt.grid()
    plt.show()

    future(model, x)
    
