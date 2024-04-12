# 実走行データで学習するためのプログラム

# ライブラリのimport
import numpy as np
import pickle
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

# データの読み込みと確認
train_data = np.loadtxt("./train_data.txt", delimiter=" ")
print("学習データのshape(サンプル数×データの項目):")
print(train_data.shape)
print("")

# 機械学習の学習データ(出力)の作成
# steer値をクラスのIDに変換(0.→0 １.→1 -1.→2)
# ・steer値が離散的なためクラス(カテゴリー)を予測する「分類」を実施
# ・steer値が連続値の場合は「回帰」→sklearn.neural_network.MLPRegressor
steer = train_data[:, 1]
class_steer = []
for i in range(steer.shape[0]):
    if steer[i] == 0.:
        class_steer.append(0)
    elif steer[i] == 1.:
        class_steer.append(1)
    else:
        class_steer.append(2)
y_train = np.array(class_steer)
print("y_trainのshape: {0}".format(y_train.shape))
print("")
# accel値を加える場合
# y_train = np.array([label_steer, label_accel]).T)

# 機械学習の学習データ(入力)の作成
# センサー値を正規化(平均0、標準偏差1)
X_train = train_data[:, 4:]
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_norm = (X_train - X_mean) / X_std
print("X_train_normのshape: {0}".format(X_train_norm.shape))
print("X_train_normの平均: {0}".format(np.mean(X_train_norm, axis=0)))
print("X_train_normの標準偏差: {0}".format(np.std(X_train_norm, axis=0)))
print("")

# 機械学習のモデル作成
# Multi-layer Perceptron classifier(多層パーセプトロン分類器)
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
model = MLPClassifier(hidden_layer_sizes=(100, 100, 100),
                     random_state=1,
                     max_iter=2000)
# X_train_norm, y_trainを使用した学習
model.fit(X_train_norm, y_train)
print("学習終了")
# 正解率の確認
print("正解率{0:.3f}".format(model.score(X_train_norm, y_train)))
print("")

# 学習済みモデルの保存
# 学習には時間がかかるため学習を終えたモデルを保存し予測時に読み込んで使用
print("セーブします。")
pickle.dump(model, open("./save_model.pickle", 'wb'))
# 予測時にセンサー値を同じく正規化するための平均・標準偏差の保存
np.savetxt("./save_mean.txt", X_mean, delimiter=" ")
np.savetxt("./save_std.txt", X_std, delimiter=" ")