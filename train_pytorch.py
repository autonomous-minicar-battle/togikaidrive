import os
import sys
import datetime
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# データの読み込みと前処理
def load_data():
    folder = "records"
    csv_files = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            csv_files.append(file)
    print(csv_files)

    if len(csv_files) > 1:
        answer = input("複数のcsvファイルがあります。ファイルを結合しますか？ (y)")
        if answer == "y":
            dataframes = []
            dataframe_colums = []
            for csv_file in csv_files:
                csv_path = os.path.join(folder, csv_file)
                df = pd.read_csv(csv_path)
                dataframes.append(df)
                dataframe_colums.append(df.columns)
                if len(dataframe_colums) > 1:
                    if not all(dataframe_colums[0] == dataframe_colums[1]):
                        print(csv_path, "の列が他のファイルと異なり結合できません。確認してください。")
                        sys.exit()
                merged_df = pd.concat(dataframes)
            df = merged_df
        else:
            csv_file = input("csvファイル名を入力してください: ")
            #csv_file = "record_20240519_224821.csv"
            csv_path = os.path.join(folder, csv_file)
            df = pd.read_csv(csv_path)
    else:
        csv_file = csv_files[0]
        csv_path = os.path.join(folder, csv_file)
        df = pd.read_csv(csv_path)

    x = df.iloc[:, 3:]
    y = df.iloc[:, 1:3]
    x_tensor = torch.tensor(x.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    print("データ形式の確認:", "x:", x_tensor.shape, "y:", y_tensor.shape)
    return x_tensor, y_tensor, csv_file

# カスタムデータセットクラス
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# モデルの定義
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# トレーニング関数
def train_model(model, dataloader, criterion, optimizer, start_epoch=0, epochs=100):
    model.train()  # モデルをトレーニングモードに設定
    for epoch in range(start_epoch, start_epoch + epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    print("トレーニングが完了しました。")
    return epoch+1
    
# 推論関数
def predict(model, x_tensor):
    model.eval()  # モデルを評価モードに設定
    with torch.no_grad():
        predictions = model(x_tensor)
    return predictions

# モデル保存関数
def save_model(model, optimizer, folder, csv_file, epoch):
    if not os.path.exists(folder):
        os.makedirs(folder)
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    model_name = f'model_{date_str}_{csv_file}_epoch_{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(folder, model_name))
    print(f"モデルを保存しました: {model_name}")

'''
# モデル読み込み関数
def load_model(model, optimizer, folder, csv_file, epoch):
    model_path = f'models/model_{csv_file}_epoch_{epoch}.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
'''
# モデル読み込み関数
def load_model(model, optimizer, folder, csv_file):
    model_files = [file for file in os.listdir(folder) if file.startswith(f'model_')]
    if model_files:
        print("利用可能なモデル:")
        print(model_files)
        model_name = input("読み込むモデル名を入力してください: ")
        model_path = os.path.join(folder, model_name)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"モデルを読み込みました: {model_name}")
        return checkpoint['epoch']
    else:
        print("利用可能なモデルが見つかりませんでした。")
        return 0

def main():
    # データのロード
    x_tensor, y_tensor, csv_file = load_data()
    
    # データセットとデータローダーの作成
    dataset = CustomDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # モデルの作成
    input_dim = x_tensor.shape[1]
    output_dim = y_tensor.shape[1]
    model = NeuralNetwork(input_dim, output_dim)
    
    # 損失関数と最適化手法の設定
    criterion = nn.MSELoss()  # 仮定: 回帰タスク
    optimizer = torch.optim.Adam(model.parameters())
        
    # モデルの読み込み
    continue_training = input("続きから学習を再開しますか？ (y/n): ").strip().lower() == 'y'
    start_epoch = 0
    
    if continue_training:
        start_epoch = load_model(model, optimizer, 'models', csv_file)
        epochs = int(input("学習するエポック数を入力してください: ").strip())

    # モデルのトレーニング
    epoch = train_model(model, dataloader, criterion, optimizer, start_epoch=start_epoch, epochs=epochs)
    
    # モデルの保存
    save_model(model, optimizer, 'models', csv_file, epoch)
    
    # 推論の実行例
    print("推論の実行例です。")
    predictions = predict(model, x_tensor)
    print(predictions)

if __name__ == "__main__":
    main()
