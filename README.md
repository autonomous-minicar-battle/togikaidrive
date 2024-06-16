# togikaidrive
超音波センサ等で自動運転するミニカーの制御プログラム
自動運転ミニカーバトルと出前授業等で活用

## 主なプログラム概要
1. motor.py　操舵・モーター出力/調整用プログラム
2. ultrasonic.py　超音波測定用プログラム
3. run.py　走行用プログラム
4. config.py　パラメータ用プログラム
5. train.py　機械学習の学習用プログラム

python run.pyで走行！それぞれのプログラムは単独チェックで活用。


## 授業
大枠
- 体験型授業ではconfigをいじるだけでできることをメインにする
- Pythonのコードは都度説明

流れ
1. 出力調整
   1. スロットル値をいじろう
   2. 舵角値をいじろう　
   ~~~
   python motor.py
   ~~~

2. 超音波センサで物との距離を測ろう
   1. 定規で距離を測り、測定値との比較をする
   2. 超音波センサの測定可能範囲（角度）を手をかざして調べる
   3. 超音波センサの数を変える
   4. サンプリングサイクルを変える
   ~~~
   python ultrasonic.py
   ~~~

3. 走行制御
   1. チキンレース！壁に直前で止まろう（パラスタ）  
   ➔config.DETECTION_DISTANCE_STOP
   2. 壁にぶつかったらバックしてみよう（追加制御）  
   ➔バック関数を作成
   3. PID制御で舵角値をいい感じにしよう（制御の改善）  
   ➔config.K_P/.K_I/.K_D
   4. MLを試そう（ルールベースの代替）  
   ➔config.mode_planとconfigのNN各種パラメータ

4. 分析
   1. 超音波センサの値を確認しよう（実測値のバラツキ、不具合）    
   ➔recordsのフォルダ
   2. 走行記録を視覚化してみよう（グラフ、画像、動画）  
   ➔imagesのフォルダ
   3. fpvで操作してみよう  
   ➔fpv.py
5. 発展
   1. IMU（加速度、ジャイロ、地磁気センサ）を使ってみよう  
   ➔gyro.py
   2. 画像処理やディープラーニングで走る
   ＊工事中

## ハードウェア
### 制限部門貸し出しマシン
![制限部門のマシン](/assets/car_onemake/car_seigenbumon.png)

#### BOM
| 分類 | 名称 | 個数 | 概算コスト(円) | 説明 |
| ---- | ---- | ---- | ---- | ---- |
| コンピュータ | [ラズパイ3B+](https://www.raspberrypi.com/products/raspberry-pi-3-model-b-plus/) |1| ---- |　販売終了
| コンピュータ | [ラズパイ3A](https://raspberry-pi.ksyic.com/main/index/pdp.id/512/pdp.open/512) |-| 5000 |（代替）|
| コンピュータ | [ラズパイ4B](https://akizukidenshi.com/catalog/g/g114839/) |-| 10000 |（代替）|
| SDカード | 配布時期による |1| ---- | 64GB以上、書き込み速度30MB/s以上推奨 |
| 距離センサ | [超音波距離センサー HC-SR04](https://akizukidenshi.com/catalog/g/g111009/) |5| 1500 | [データシート](https://akizukidenshi.com/goodsaffix/hc-sr04_v20.pdf)
| ジャイロ加速度センサ | [BNO055使用 9軸センサーフュージョンモジュールキット](https://akizukidenshi.com/catalog/g/g116996/) |任意| 2500 | [データシート](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bno055-ds000.pdf)
| カメラ | [ラズベリー•パイ（Raspberry Pi）160°広角500MP](https://jp.sainsmart.com/products/wide-angle-fov160-5-megapixel-camera-module-for-raspberry-pi) |任意| 3000 |　コース内特徴を捉えるため、広角推奨。
| シャーシ | [RCスターユニット 1/14 エアロ アバンテ](https://tamiyashop.jp/shop/g/g57402/) |1| 6500 |　販売終了
| モーター | シャーシに含む |1| ---- |
| コンピュータ用バッテリ | [Anker PowerCore Fusion 5000](https://amzn.asia/d/b78Zim4) |1| 3600 |
| 駆動用バッテリ | [単３電池]() |4| 400 |
| モータドライバ | [RC ESC 20A ブラシモーター](https://www.amazon.co.jp/GoolRC-%E3%83%96%E3%83%A9%E3%82%B7%E3%83%A2%E3%83%BC%E3%82%BF%E3%83%BC-%E3%82%B9%E3%83%94%E3%83%BC%E3%83%89%E3%82%B3%E3%83%B3%E3%83%88%E3%83%AD%E3%83%BC%E3%83%A9%E3%83%BC-%E5%88%87%E3%82%8A%E6%9B%BF%E3%81%88%E5%8F%AF%E8%83%BD-%E3%83%96%E3%83%AC%E3%83%BC%E3%82%AD%E4%BB%98/dp/B014RB6WS6) |1| 1500 |
| サーボドライバ | [PCA9685 16チャンネル 12-ビット PWM Servo モーター ドライバー](https://amzn.asia/d/0sswysQ) |1| 1000 |
| コントローラー | [Logicool G ゲームパッド コントローラー F710](https://www.amazon.co.jp/%E3%83%AD%E3%82%B8%E3%82%AF%E3%83%BC%E3%83%AB-F710r-%E3%80%90%E3%83%A2%E3%83%B3%E3%82%B9%E3%82%BF%E3%83%BC%E3%83%8F%E3%83%B3%E3%82%BF%E3%83%BC%E3%83%95%E3%83%AD%E3%83%B3%E3%83%86%E3%82%A3%E3%82%A2%E6%AD%A3%E5%BC%8F%E6%8E%A8%E5%A5%A8%E3%80%91-LOGICOOL-%E3%83%AF%E3%82%A4%E3%83%A4%E3%83%AC%E3%82%B9%E3%82%B2%E3%83%BC%E3%83%A0%E3%83%91%E3%83%83%E3%83%89/dp/B00CDG7994) |1| 4000 |
| 締結部品 | [2mm六角スペーサ](https://www.amazon.co.jp/%E3%83%8A%E3%82%A4%E3%83%AD%E3%83%B3%E3%83%8D%E3%82%B8%E3%83%8A%E3%83%83%E3%83%88-320%E5%80%8B%E3%82%BB%E3%83%83%E3%83%88-%E5%85%AD%E8%A7%92%E3%82%B9%E3%83%9A%E3%83%BC%E3%82%B5%E3%83%BC-%E3%82%B9%E3%82%BF%E3%83%B3%E3%83%89%E3%82%AA%E3%83%95-%E5%8F%8E%E7%B4%8D%E3%82%B1%E3%83%BC%E3%82%B9%E4%BB%98%E3%81%8D/dp/B09G9RPC18/ref=sr_1_34_sspa?dib=eyJ2IjoiMSJ9.v2Z5JMko630Hc7v-Db1vOLYgTcYCkoMUhfz5IF_I-4JzqykRRxcumS9lJH4CKRcZ15qY-ViSoY3mtOiVZ0QP2wZkjw5S2E_UsbHvFKbaAgUxhOZUDZnY04JrS-doS5FGCc5ihOEbmM6H6voaFzNCjI46_wAnwlSwjeBHu8YuoFJTpUrYDTPbYk2T87zNKMDjfvW7avb-M0O-T4HuXnUi2xE98TZeNuB1jUJXaeh3tX3x7mQEx-yJYUpk9ZUcs2HSCpgzlfMUIAT36_JyIaXNXcYC9brXbkFmLpu3ATNf_Po.wq0WsIwMoUsaMbQw_f9EKbe3EONGyw4YZiOi3AQ8UR8&dib_tag=se&keywords=6%E8%A7%92%E3%82%B9%E3%83%9A%E3%83%BC%E3%82%B5%E3%83%BC+2mm&qid=1713080306&sr=8-34-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9tdGY&psc=1) |16|1000 |ラズパイマウント用 |
| 締結部品 | 2mm六角スペーサ |6| ↑|サーボドライバ用 |
| マウント | ラズパイ/バッテリマウント |1|1000 | 材料費のみ換算
| マウント | カメラマウント |1| 300| 材料費のみ換算 |
| マウント | 超音波センサマウント |1| 500| 材料費のみ換算 |
| ケーブル | ジャンパワイヤ　メス-オス |5| 1000| 超音波センサ用 |
| ケーブル | ジャンパワイヤ　メス-メス |1| 1000| サーボドライバ用 |
| togikai基盤 | サーボドライバ代替 |任意| --- |  HC-SR04*8個接続用ジャンパピン・ PCA9685 2ch・OLED・ファン電源搭載、秋月BNO055モジュール追加搭載用I2Cスルーホール有 |

#### 組み立てマニュアル
＊工事中
#### 環境構築
1. 利用するOSは[2021-01-11-raspios-buster-i386.iso](https://downloads.raspberrypi.com/rpd_x86/images/rpd_x86-2021-01-12/2021-01-11-raspios-buster-i386.iso)  
donkeycar 4.4.0を利用し約するため、busterを採用。

2. Raspberry [Pi Imager](https://www.raspberrypi.com/software/)を使ってSDカードへ書き込み

3. [togikaidrive](https://github.com/autonomous-minicar-battle/togikaidrive.git)をgit cloneする
   ~~~
   git clone https://github.com/autonomous-minicar-battle/togikaidrive.git
   ~~~
4. パスワードなしSSHログイン：[参考](https://qiita.com/Ash_root/items/143f7f21373f43127da6)

5. wifiの設定ファイル設置と暗号化：[参考](https://raspida.com/wifisetupfile/)
   1. 過去wifi設定ミスっているやつでつながらない場合：
     [参考](https://tm-progapp.hatenablog.com/entry/2022/03/30/112529)

6. メモリが少ないのでswapを増やす：
   [参考](https://nekopom.jp/raspberrypi_setting09/#index_id0)

7. ライブラリ類
   1. [OpenCV](https://opencv.org/)
      ~~~ 
      sudo apt install python3-opencv
      ~~~ 
   2. [Flask](https://msiz07-flask-docs-ja.readthedocs.io/ja/latest/)
      ~~~ 
      pip install Flask
      ~~~ 
   3. [Pytorch](https://pytorch.org/)
      ビルドからやると大変（でした）なので、先人のをありがたく使います。  
      参考：https://zenn.dev/kotaproj/articles/c10c5cb3a03c52
      ~~~ 
      sudo apt update
      sudo apt upgrade
      sudo apt install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools
      sudo apt install libatlas-base-dev
      git clone https://github.com/Kashu7100/pytorch-armv7l.git
      cd pytorch-armv7l/
      pip install torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
      pip install torchvision-0.8.0a0+45f960c-cp37-cp37m-linux_armv7l.whl
      ~~~
      エラーがないことを確認
      ~~~
      $ python
      >>> import torch
      >>> import torchvision
      >>> torch.__version__
      '1.7.0a0+e85d494'
      >>> torchvision.__version__
      '0.8.0a0+45f960c'
      >>> exit()      
      ~~~
   4. [Adafruit_PCA9685](https://github.com/adafruit/Adafruit_Python_PCA9685)
      モーターを動かすのに使います。  
      ~~~
      pip install Adafruit_PCA9685
      ~~~
   5. pygame コントローラーを使うときに使います。
      ~~~
      pip install pygame
      ~~~
   
   5. ジャイロに挑戦する方はインストール　たくさん種類があります。  

      - [BNO055使用 9軸センサーフュージョンモジュールキット](https://akizukidenshi.com/catalog/g/g116996/)、togikai基盤にそのまま乗ります。  
      [参考](https://github.com/ghirlekar/bno055-python-i2c) 
      ~~~
      sudo nano /etc/modules
      ~~~
      下記を追記  
      i2c-bcm2708  
      i2c-dev

      モジュールとプログラムをインストール
      ~~~
      sudo apt-get install python-smbus
      sudo apt-get install i2c-tools
      git clone https://github.com/ghirlekar/bno055-python-i2c.git
      ~~~
      i2cの接続確認テスト
      ~~~
      sudo i2cdetect -y 1
      ~~~
      サンプルプログラムの実行
      ~~~
      cd bno055-python-i2c
      python BNO055.py
      ~~~

      - 他



   

## ツール類
   - エディター：[VS Code](https://code.visualstudio.com/) 
      - [GitHub Copilot](https://github.com/github/copilot-preview) AIによるコード補完機能で、プログラミング作業を効率化する拡張機能。
      - [SFTP](https://marketplace.visualstudio.com/items?itemName=Natizyskunk.sftp)ファイル転送用拡張機能。[参考](https://note.com/_nakashimmer_/n/nd10a5acc6f43)

- コード管理：[Git](https://git-scm.com/) 
- コード配布：[GitHub](https://github.com/) 
 - GUIでファイル転送：[Filezilla](https://filezilla-project.org/)
