import os
import sys
import datetime
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config

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
    print("入力データのヘッダー確認:\n", df.columns)
    print("データの確認:\n", df.head())

    #df.iloc[:, 1:] = df.iloc[:, 1:].astype(int)
    x = df.iloc[:, 3:]
    y = df.iloc[:, 1:3]
    x_tensor = torch.tensor(x.values, dtype=torch.float32)
    x_tensor =x_tensor / 2000 # 2000mmを1として正規化
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    y_tensor = y_tensor / 100 # 100%を1として正規化
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
    def __init__(self, input_dim, output_dim,hidden_dim, num_hidden_layers):
        super(NeuralNetwork, self).__init__()
        ## 変数で層を追加
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*layers)

        # 手動で層を追加
        #self.layers = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, output_dim)
        #)

    def forward(self, x):
        x = self.layers(x)
        return x

    # 推論関数
    def predict(self, model, x_tensor):
        model.eval()  # モデルを評価モードに設定
        with torch.no_grad():
            predictions = model(x_tensor)
            predictions = F.softmax(predictions, dim=1)
        return predictions

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
    

# モデル保存関数
def save_model(model, optimizer, folder, csv_file, epoch):
    if not os.path.exists(folder):
        os.makedirs(folder)
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    model_name = f'model_{date_str}_{csv_file}_epoch_{epoch}_{config.ultrasonics_list_join}.pth'
    model_path = os.path.join(folder, model_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)
    print(f"モデルを保存しました: {model_path}")
    return model_path


def load_model(model, model_path=None,optimizer=None, folder='.'):
    """
    モデルを指定したフォルダーから読み込む関数。
    
    Args:
    - model: PyTorchモデルのインスタンス
    - optimizer: PyTorchのoptimizerのインスタンス（省略可能）
    - folder: モデルファイルが保存されているフォルダー

    Returns:
    - epoch: 読み込んだモデルのエポック数（失敗時は0）
    """
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("オプティマイザの状態も読み込みました。")
        print(f"モデルを読み込みました: {model_path}")
        return checkpoint.get('epoch', 0)
    else:
        model_files = [file for file in os.listdir(folder) if file.startswith('model_')]
        if model_files:
            print("利用可能なモデル:")
            print(model_files)
            model_name = input("読み込むモデル名を入力してください.\n！注意！過去にモデル構造を変更している場合は読み込めませんので、config.pyを編集してください。\n: ")
            model_path = os.path.join(folder, model_name)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("オプティマイザの状態も読み込みました。")
            print(f"モデルを読み込みました: {model_name}")
            return checkpoint.get('epoch', 0)
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
    model = NeuralNetwork(input_dim, output_dim, config.hidden_dim, config.num_hidden_layers)
    
    # 損失関数と最適化手法の設定
    criterion = nn.MSELoss()  # 仮定: 回帰タスク
    optimizer = torch.optim.Adam(model.parameters())
        
    # モデルの読み込み
    continue_training = input("続きから学習を再開しますか？ (y): ").strip().lower() == 'y'
    start_epoch = 0
    
    if continue_training:
        start_epoch = load_model(model, None, optimizer, 'models')
    else: start_epoch =0
    try: epochs = int(input("学習するエポック数を入力してください.(デフォルト:100): ").strip())
    except ValueError: epochs = 100
    
    # モデルのトレーニング
    epoch = train_model(model, dataloader, criterion, optimizer, start_epoch=start_epoch, epochs=epochs)
    
    # モデルの保存
    model_path = save_model(model, optimizer, 'models', csv_file, epoch)
    
    # 推論の実行例
    ## NNモデルの読み込み
    #model = NeuralNetwork(input_dim, output_dim)
    model_dir = "models"
    #model_name = config.model_name #"model_20240527_record_20240519_224821.csv.pth"
    #model_path = os.path.join(model_dir, model_name)
 
   # 保存したモデルを再度ロード
    print("\n保存したモデルを再度ロードします。")
    load_model(model, model_path, None, model_dir)
    print(model)
 
    print("\n推論の実行例です。")
    input_example = x_tensor[:4]
    predictions = model.predict(model, input_example)
    print("\n入力データ:")
    print(input_example)
    print("\n予測結果:")
    print(predictions)

if __name__ == "__main__":
    main()
