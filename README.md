# togikaidrive
走行用のメインプログラム

## タスク（事務局用）
1. gyro追加（済
2. 可視化（操作ｖｓセンサ値、カメラ画像）(fpv作成)
3. MLをPytorchに書き換え（train、test）（小野寺さん
4. 画像処理、CNNコンテンツ追加（小野寺さん、実行委員）
5. 2Dシミュレータ@colab（東松さん）

細かいところ今後修正予定
- recordではなく、logとする
- configはpythonではなく、jsonにする
- motorでpigpiodを使う
- logファイル（記録の可視化、カメラ画像）
- ultrasonicのフィルタ


## プログラム概要(この順番で教える)
1. motor.py　操舵・モーター出力/調整用プログラム
2. ultrasonic.py　超音波測定用プログラム
3. run.py　走行用プログラム
4. config.py　パラメータ用プログラム


## 授業の流れ
- チャレンジしてみること（configをいじるだけでできること）
- Pythonのコードは都度説明
1. 出力調整
➔motor.py
   1. スロットル値をいじろう
   2. 舵角値をいじろう　

2. 超音波センサで物との距離を測ろう
➔ultrasonic.py
   1. 定規で距離を測り、測定値との比較をする
   2. 超音波センサの測定可能範囲（角度）を手をかざして調べる
   3. 超音波センサの数を変える
   4. サンプリングサイクルを変える

3. 走行制御
   1. チキンレース！壁に直前で止まろう（パラスタ）
   ➔config.DETECTION_DISTANCE_STOP
   2. 壁にぶつかったらバックしてみよう（追加制御）
   ➔バック関数を作成
   3. PID制御で舵角値をいい感じにしよう（制御の改善）
   ➔config.mode_planとconfigのNN各種パラメータ
   4. MLを試そう（ルールベースの代替）
   ➔config.K_P/.K_I/.K_D
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
| シャーシ | [RCスターユニット 1/14 エアロ アバンテ](https://tamiyashop.jp/shop/g/g57402/) |1| 6500 |
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
| togikai基盤 | --- |任意|  HC-SR04*8個接続ピン・ PCA9685 2ch・OLED搭載、秋月BNO055モジュール追加搭載可能 |

#### 組み立てマニュアル
　ココに入れる
#### ラズパイのセットアップ
1. OS:[2021-01-11-raspios-buster-i386.iso](https://downloads.raspberrypi.com/rpd_x86/images/rpd_x86-2021-01-12/2021-01-11-raspios-buster-i386.iso)
2. Raspberry [Pi Imager](https://www.raspberrypi.com/software/)を使ってSDカードへ書き込み
3. [togikaidrive](https://github.com/autonomous-minicar-battle/togikaidrive.git)をgit cloneする
4. パスワードなしSSHログイン：[参考](https://qiita.com/Ash_root/items/143f7f21373f43127da6)
5. ライブラリ類
   1. [OpenCV](https://opencv.org/)
      1. sudo apt install python3-opencv
   2. [Flask](https://msiz07-flask-docs-ja.readthedocs.io/ja/latest/)
      1. pip install Flask
   

## ツール類
### エディター

   - エディター：[VS Code](https://code.visualstudio.com/) 
      - [GitHub Copilot](https://github.com/github/copilot-preview) AIによるコード補完機能で、プログラミング作業を効率化する拡張機能。
      - [SFTP](https://marketplace.visualstudio.com/items?itemName=Natizyskunk.sftp)ファイル転送用拡張機能。[参考](https://note.com/_nakashimmer_/n/nd10a5acc6f43)

- コード管理：[Git](https://git-scm.com/) 
- コード配布：[GitHub](https://github.com/) 
 - GUIでファイル転送：[Filezilla](https://filezilla-project.org/)

### HP作成
- WEBデザイン作成：[Figma](https://www.figma.com/) 

   オンラインのデザインツールです。UI/UXデザインやプロトタイプ作成に使用されます。Figmaを使用すると、複数のユーザーが同時に作業できるため、チームでの協力が容易になります。

- WEBサイト作成ノーコードツール：[Studio](https://studio.design/ja) 

   Figmaで作ったデザインをインポートする。[参考](https://studio.design/ja/figma-to-studio)
- ヘッドレスCMS：[microCMS](https://microcms.io/) 

   microCMSはAPIベースのヘッドレスCMSです。StudioとAPI連携させてDB管理。今後の拡張用のために採用。

#### 作成の流れ
   Figmaでデザイン作成➔StudioでWEBフレームワークに変換し、公開。
   DBの管理はmicroCMSを使う。　昨年までStudioだけで作成していたが、デザイン修正が大変であったためFigmaを、DBの拡張性のためmicroCMSを使いたい。
   
