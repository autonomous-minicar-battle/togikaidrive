import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# "record"フォルダ内の全てのcsvファイルを取得
folder = "records"
print("Training: type file name containing data below:" )
csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

# 各csvファイルを読み込み、データフレームに格納
dataframes = [pd.read_csv(os.path.join(folder, f)) for f in csv_files]

# 1,2の列が目標変数
X = pd.concat([df.iloc[:, 2:] for df in dataframes])
y = pd.concat([df.iloc[:, 1:2] for df in dataframes])

# 各特徴量の平均が0、標準偏差が1になるようにデータを正規化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# データをテンソルに変換
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

# データセットとデータローダーを作成
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32)

# 3層のニューラルネットワークを定義
model = nn.Sequential(
    nn.Linear(X.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2)  
)

# 損失関数と最適化手法を定義
criterion = nn.MSELoss()  # 仮定: 回帰タスク
optimizer = torch.optim.Adam(model.parameters())

# 学習ループ
for epoch in range(100):  # 100エポックで学習
    for inputs, targets in dataloader:
        # 前方計算
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 勾配をゼロにリセット
        optimizer.zero_grad()

        # 逆伝播計算
        loss.backward()

        # パラメータ更新
        optimizer.step()

# モデルを保存するディレクトリを作成
if not os.path.exists('models'):
    os.makedirs('models')

# 学習完了時の日付を取得
date_str = datetime.now().strftime('%Y%m%d')

# 学習に使用したCSVファイルの名前を取得
csv_names = '_'.join([os.path.splitext(f)[0] for f in csv_files])

# モデルを保存
model_name = f'model_{date_str}_{csv_names}.pth'
torch.save(model.state_dict(), os.path.join('models', model_name))