from train_pytorch import load_data, CustomDataset, NeuralNetwork,  test_model
import os
import sys
import time

import config
 

def main():
    # データのロード
    x_tensor, y_tensor, csv_file = load_data()
    
    # データセットとデータローダーの作成
    dataset = CustomDataset(x_tensor, y_tensor)
    #dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # モデルの作成
    input_dim = x_tensor.shape[1]
    output_dim = y_tensor.shape[1]
    model = NeuralNetwork(input_dim, output_dim, config.hidden_dim, config.num_hidden_layers)
            
    # モデルのテスト
    print("モデルのテストを開始します...")
    print("モデルのパス: ", config.model_path)
    answer = input("\n上記のモデル以外でテストしますか？ (y)")
    if answer == "y":
        folder = "models"
        models = []
        for m in os.listdir(folder):
            if m.endswith(".pth"):
                models.append(m)
        print(models)
        if len(models) > 1:
            answer = input("\nテストするモデル名を入力してください, Etrで最後を選択: ")
            if answer == "":
                answer = models[-1]
                print("\n最新のファイルを選択：",answer)
                time.sleep(0.5)
            config.model_path = os.path.join(folder, answer)
        else:
            print("モデルが見つかりませんでした。")
            sys.exit()
    else:
        pass
    test_model(model, config.model_path,dataset,x_tensor.shape[0])
 
if __name__ == "__main__":
    main()
