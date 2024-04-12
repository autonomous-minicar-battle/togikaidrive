# 学習結果を確認するプログラム

# ライブラリのimport
import numpy as np
import pickle
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

# 学習済みモデルの読み込み
model = pickle.load(open("./save_model.pickle", 'rb'))
# センサー値を学習時と同じく正規化するため平均・標準偏差の読み込み
X_mean = np.loadtxt("./save_mean.txt", delimiter=" ")
X_std = np.loadtxt("./save_std.txt", delimiter=" ")

# テストデータ(精度を確認するためのデータ)の読み込み
# ・テストデータは学習データと異なるものを使用
# ・学習データの精度に対しテストデータの精度が大きく低下する場合は「過学習」の可能性
test_data = np.loadtxt("./test_data.txt", delimiter=" ")
# センサー値
X_test = test_data[:, 4:]
# 正規化
X_test_norm = (X_test - X_mean) / X_std

# 予測
pred = model.predict(X_test_norm)

# 予測で出力されるクラスID(0, 1, 2)をsteer値(0., 1., -1.)に変換
pred_steer = []
for i in range(pred.shape[0]):
    if pred[i] == 0:
        pred_steer.append(0.)
    elif pred[i] == 1:        
        pred_steer.append(1.)
    else:
        pred_steer.append(-1.)        
pred_steer = np.array(pred_steer)

# 予測精度を確認するためのsteer値の正解データ
steer = test_data[:, 1]

# steer値をクラスのIDに変換(0.→0 １.→1 -1.→2)
class_steer = []
for i in range(steer.shape[0]):
    if steer[i] == 0.:
        class_steer.append(0)
    elif steer[i] == 1.:
        class_steer.append(1)
    else:
        class_steer.append(2)
y_test = np.array(class_steer)

# 正解率の確認
print("正解率{0:.3f}".format(model.score(X_test_norm, y_test)))
# 混同行列の確認
# ・各クラスのデータがどのクラスに予測されたかを示す表
# ・正解率だけではわからない情報が把握可能
confusion_matrix = metrics.confusion_matrix(y_test, pred)
print("")
print("混同行列")
print("       {0:>6s}{1:>6s}{2:>6s}".format("予測 0", "予測 1", "予測-1"))
print("      " + "-" * 23)
print("正解 0 |{0[0]:7d}{0[1]:7d}{0[2]:7d}".format(confusion_matrix[0]))
print("正解 1 |{0[0]:7d}{0[1]:7d}{0[2]:7d}".format(confusion_matrix[1]))
print("正解-1 |{0[0]:7d}{0[1]:7d}{0[2]:7d}".format(confusion_matrix[2]))