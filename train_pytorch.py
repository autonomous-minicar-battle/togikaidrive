import os
import time
import sys
import datetime
print("ライブラリの初期化に数秒かかります...")
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader #, TensorDataset
import matplotlib.pyplot as plt

import config

if config.mode_plan == "CNN": 
    from PIL import Image
    from torchvision import transforms
    import torch.optim as optim


# データの読み込みと前処理
def load_data():
    folder = "records"
    csv_files = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            csv_files.append(file)
    print(csv_files)

    if len(csv_files) > 1:
        answer = input("\n複数のcsvファイルがあります。ファイルを結合しますか？ (y)")
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
            csv_file = input("ファイル名を入力してください, Etrで最後を選択: ")
            if csv_file == "":
                csv_file = csv_files[-1]
                print("\n最新のファイルを選択：",csv_file)
                time.sleep(0.5)
            #csv_file = "record_20240519_224821.csv"
            csv_path = os.path.join(folder, csv_file)
            df = pd.read_csv(csv_path)
    else:
        csv_file = csv_files[0]
        csv_path = os.path.join(folder, csv_file)
        df = pd.read_csv(csv_path)
    #print("\n入力データのヘッダー確認:\n", df.columns)
    print("\n入力データの確認:\n", df.head(3), "\nデータサイズ",df.shape)

    # データの前処理
    x = df.iloc[:, 3:]
    x_tensor = torch.tensor(x.values, dtype=torch.float32)
    x_tensor = normalize_ultrasonics(x_tensor)

    if config.model_type == "categorical":
        # -100から100の範囲をconfig.num_categoriesの数に分割してカテゴリに変換
        df['Str'] = pd.cut(df['Str'], bins=config.bins_Str, labels=False)
        #df['Thr'] = pd.cut(df['Thr'], bins=config.bins_Thr, labels=False)
        y = df.iloc[:, 1:3]
        y_tensor = torch.tensor(y.values, dtype=torch.long)  # long型のテンソルに変更
        print("\n学習データの確認:\n教師データ(ラベル：Str, Thr[学習なし]):", y_tensor[0,:], "\n入力データ(正規化センサ値)：", x_tensor[0,:])
        print("学習データサイズ:", "y:", y_tensor.shape, "x:", x_tensor.shape, "\n")

    else:
        y = df.iloc[:, 1:3]
        y_tensor = torch.tensor(y.values, dtype=torch.float32)
        y_tensor = normalize_motor(y_tensor)
        y_tensor[:, 0] = steering_shifter_to_01(y_tensor[:, 0])  # ステアリングの値を-1~1を0~1に変換
        print("\n学習データの確認:\n教師データ(正規化操作値+0.5: Str, Thr):", y_tensor[0,:], "\n入力データ(正規化センサ値)：", x_tensor[0,:])
        print("学習データサイズ:", "y:", y_tensor.shape, "x:", x_tensor.shape, "\n")

    #y_tensor = steering_shifter_to_01(y_tensor) # -1~1を0~1に変換

    # timestampの取得
    ts = df.iloc[:, 0]

    return x_tensor, y_tensor, csv_file, ts

def normalize_ultrasonics(x_tensor, scale=2000):
    x_tensor = x_tensor / scale # 2000mmを1として正規化
    return x_tensor

def denormalize_ultrasonics(x_tensor, scale=2000):
    x_tensor = x_tensor * scale # 2000mmを1として正規化
    return x_tensor

def normalize_motor(y_tensor, scale=100):
    y_tensor = y_tensor / scale # 100%を1として正規化
    return y_tensor

def denormalize_motor(y_tensor, scale=100):
    y_tensor = y_tensor * scale # 100%を1として正規化
    return y_tensor

# -1~1を0~1に変換
def steering_shifter_to_01(y_tensor):
    y_tensor = (y_tensor +1 )/2
    return y_tensor

# 0~1を-1~1に変換
def steering_shifter_to_m11(y_tensor):
    y_tensor = (y_tensor-0.5)*2
    return y_tensor

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
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        if config.model_type == "categorical":
            layers.append(nn.Linear(hidden_dim, config.num_categories))
        else:
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if config.model_type == "categorical":
            x = F.log_softmax(x, dim=1)
        return x

    def predict(self, model, x_tensor):
        model.eval()  # モデルを評価モードに設定
        with torch.no_grad():
            predictions = model(x_tensor)
            if config.model_type == "categorical":
                predictions = torch.argmax(predictions, dim=1)
                #print("predictions:",predictions)
                predictions[0] = config.categories_Str[predictions[0]]/100
                #predictions[1] = config.categories_Str[predictions[1]]/100
                predictions = torch.tensor([[predictions[0],config.categories_Thr[predictions[0]]/100]])
            else: predictions[:,0] = steering_shifter_to_m11(predictions[:,0]) # 0~1を-1~1に変換
        predictions = torch.clamp(predictions, min=-1, max=1)  # Clamp values between -1 and 1
        return predictions

    def predict_label(self, model, x_tensor):
        model.eval()  # モデルを評価モードに設定
        with torch.no_grad():
            predictions = model(x_tensor)
            predictions = torch.argmax(predictions, dim=1)
            #print("predictions:",predictions)
        return predictions

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, y_tensor, ts, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.y_tensor = y_tensor
        self.image_paths = []
        self.labels = []

        # タイムスタンプをキーとしてインデックスをマップ
        #self.tstamp_to_index = {tstamp: i for i, tstamp in enumerate(df['Tstamp'])}
        self.tstamp_to_index = {tstamp: i for i, tstamp in enumerate(ts)}

        print("画像パスとラベルを設定中...")
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for img_name in os.listdir(subdir_path):
                    tstamp = float(img_name.split('_')[0])  # 例：1718996914.99_100_100.jpgからタイムスタンプを取得
                    if tstamp in self.tstamp_to_index:
                        self.image_paths.append(os.path.join(subdir_path, img_name))
                        index = self.tstamp_to_index[tstamp]
                        self.labels.append(y_tensor[index])

        print(f"画像パスとラベルの設定完了。画像数: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = self.labels[idx]
        return image, labels


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, 
                 input_dim = (config.IMAGE_W, config.IMAGE_H, config.IMAGE_DEPTH),
                 #input_dim: Tuple[int, ...] = (120, 160, 3)
                 output_dim = 2
                 ):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        #self.pool = nn.MaxPool2d(2, stride=2)

        # channel,conved channel,filter(kernel) size
        self.conv1 = nn.Conv2d(3,24,5,stride=2)
        self.conv2 = nn.Conv2d(24,32,5,stride=2)
        self.conv3 = nn.Conv2d(32,64,5,stride=2)
        self.conv4 = nn.Conv2d(64,64,3,stride=1)
        self.conv5 = nn.Conv2d(64,64,3,stride=1)
        self.layer1 = nn.Sequential(
            self.conv1, self.relu, self.dropout,
            self.conv2, self.relu, self.dropout,
            self.conv3, self.relu, self.dropout,
            self.conv4, self.relu, self.dropout,
            self.conv5, self.relu, self.dropout,
        )

        n_size = self._get_conv_output(input_dim)
        print("Size of CV2FC: ",n_size)

        #self.fc1 = nn.Linear(64 * 5 * 5, 100)
        self.fc1 = nn.Linear(n_size, 100)
        self.fc2 = nn.Linear(100, 50)      
        self.fc3 = nn.Linear(50, output_dim)
        self.layer2 = nn.Sequential(
            self.fc1, self.relu, self.dropout,
            self.fc2, self.relu, self.dropout,
            self.fc3
        )

    def _get_conv_output(self, shape):
        print("shape: ",shape[0],shape[1],shape[2])
        input = torch.autograd.Variable(torch.rand(1, 3, shape[1], shape[0]))
        output_feat = self.layer1(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def forward(self, x):
            x = self.layer1(x)
            x = x.view(x.size()[0], -1)
            x = self.layer2(x)
            return x

    def predict(self, model, x_tensor):
        model.eval()  # モデルを評価モードに設定
        with torch.no_grad():
            predictions = model(x_tensor)
            if config.model_type == "categorical":
                predictions = torch.argmax(predictions, dim=1)
                #print("predictions:",predictions)
                predictions[0] = config.categories_Str[predictions[0]]/100
                #predictions[1] = config.categories_Str[predictions[1]]/100
                predictions = torch.tensor([[predictions[0],config.categories_Thr[predictions[0]]/100]])
            else: predictions[:,0] = steering_shifter_to_m11(predictions[:,0]) # 0~1を-1~1に変換
        predictions = torch.clamp(predictions, min=-1, max=1)  # Clamp values between -1 and 1
        return predictions

    def predict_label(self, model, x_tensor):
        model.eval()  # モデルを評価モードに設定
        with torch.no_grad():
            predictions = model(x_tensor)
            predictions = torch.argmax(predictions, dim=1)
            #print("predictions:",predictions)
        return predictions


# トレーニング関数
def train_model(model, dataloader, criterion, optimizer, model_name, start_epoch=0, epochs=config.epochs):
    model.train()  # モデルをトレーニングモードに設定
    loss_history = []  # Loss values for plotting
    for epoch in range(start_epoch, start_epoch + epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if config.model_type == "categorical":
                targets = targets[:,0] #.squeeze(dim=-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        loss_history.append(loss.item())  # Record the loss value
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    print("トレーニングが完了しました。")
    
    # Plot and save the loss values
    plt.figure()
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_history_path = config.model_dir+'/'+ model_name +'loss_history.png'
    plt.savefig(loss_history_path)
    plt.close()
    print("Lossの履歴を保存しました: "+loss_history_path)
    return epoch+1
    

# モデル保存関数
#def save_model(model, optimizer, folder, csv_file, epoch):
def save_model(model, optimizer, folder, model_name, epoch):
    if not os.path.exists(folder):
        os.makedirs(folder)
    model_path = os.path.join(folder, model_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)
    print(f"モデルを保存しました: {model_name}")
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
    

def test_model(model, model_path, dataset,sample_num=5):
   # 推論の実行例
    ## NNモデルの読み込み
    #model = NeuralNetwork(input_dim, output_dim)
    model_dir = "models"
    #model_name = config.model_name #"model_20240527_record_20240519_224821.csv.pth"
    #model_path = os.path.join(model_dir, model_name)
 
   # 保存したモデルを再度ロード
    print("\n保存したモデルをロードします。")
    load_model(model, model_path, None, model_dir)
    print(model)
 
    print("\n推論の実行例です。\nランダムに",sample_num,"コのデータを取り出して予測します。")
    # dataの取り出し
    testloader = DataLoader(dataset, batch_size=1, shuffle=True)
    tmp = testloader.__iter__()
    x = torch.tensor([])
    y = torch.tensor([])
    yh = torch.tensor([])
    for _ in range(sample_num):
        x1, y1 = next(tmp) # 1バッチ分のデータを取り出す
        x = torch.cat([x, x1])
        y = torch.cat([y, y1])
        if config.model_type == "linear": 
            yh1 = model.predict(model, x1)
            yh = torch.cat([yh, yh1])
        elif config.model_type == "categorical":
            yh1 = model.predict_label(model, x1)     
            yh = torch.cat([yh, torch.tensor([yh1, config.categories_Str[yh1]]).unsqueeze(0) ])
                
    print("\n入力データ:")
    print(x)
    print("\n正解データ:")
    print(y)
    print("\n予測結果:")
    print(yh)
    if config.model_type == "categorical":
        print("\n正解率_Str: ", 
              int(torch.sum(y[:,0] == yh[:,0]).item()/sample_num*100),"%")
        print("confusion matrix_Str:\n",
              pd.crosstab(y[:,0], yh[:,0], rownames=['True'], colnames=['Predicted'], margins=True))
        print("\n正解率_Thr: ", 
              int(torch.sum(y[:,1] == yh[:,1]).item()/sample_num*100),"%")

    print("\n使用したモデル名：",os.path.split(model_path)[-1])


def main():
    # データのロード
    x_tensor, y_tensor, csv_file, ts = load_data()
    
    # データセットとデータローダーの作成, モデルの作成
    if config.mode_plan == "NN":
        dataset = CustomDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        input_dim = x_tensor.shape[1]
        output_dim = y_tensor.shape[1]
        model = NeuralNetwork(input_dim, output_dim, config.hidden_dim, config.num_hidden_layers)
    elif config.mode_plan == "CNN":
        transform = transforms.Compose([
            transforms.Resize((160, 120)),
            transforms.ToTensor(),
        ])
        dataset = CustomImageDataset(root_dir="images", y_tensor=y_tensor, ts=ts, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        input_dim = (config.IMAGE_W, config.IMAGE_H, config.IMAGE_DEPTH)
        output_dim = y_tensor.shape[1]
        model = ConvolutionalNeuralNetwork(input_dim, output_dim)
    else: 
        print("学習モデルが選択されていません。config.pyのmode_planを設定してください。")
        sys.exit()
    print("モデル構造: ",model)

    # 損失関数と最適化手法の設定
    if config.model_type == "categorical":
        criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss for classification
    else:
        criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters())
        
    # モデルの読み込み
    continue_training = input("続きから学習を再開しますか？ (y): ").strip().lower() == 'y'
    start_epoch = 0
    
    if continue_training:
        start_epoch = load_model(model, None, optimizer, 'models')
    else: start_epoch =0
    try: epochs = int(input(f"学習するエポック数を入力してください.(デフォルト:{config.epochs}): ").strip())
    except ValueError: epochs = config.epochs
    

    # モデルの保存名
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    model_name = f'model_{date_str}_{csv_file}.pth'.replace(".csv","")

    # モデルのトレーニング
    #epoch = train_model(model, dataloader, criterion, optimizer, start_epoch=start_epoch, epochs=epochs)
    epoch = train_model(model, dataloader, criterion, optimizer, model_name, start_epoch=start_epoch, epochs=epochs)
    
    # モデルの保存
    #model_path = save_model(model, optimizer, 'models', csv_file, epoch)
    model_path = save_model(model, optimizer, 'models', model_name, epoch)
    
    # モデルのテスト
    test_model(model, model_path,dataset)
 
if __name__ == "__main__":
    main()
