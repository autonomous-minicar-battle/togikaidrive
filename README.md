# togikaidrive
走行用のメインプログラム

## タスク
1. runを書く　済
2. PIDをいれる　済
3. MLをPytorchに書き換え（train、test）
4. 可視化（操作ｖｓセンサ値、カメラ画像）jupyternotebook用意する
5. 画像処理、CNNコンテンツ追加
6. 

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
   1. スロットル値をいじろう
   2. 舵角値をいじろう　

2. 超音波センサで物との距離を測ろう
   1. 定規で距離を測り、測定値との比較をする
   2. 超音波センサの測定可能範囲（角度）を手をかざして調べる
   3. 超音波センサの数を変える
   4. サンプリングサイクルを変える

3. 走行制御
   1. チキンレース！壁に直前で止まろう（パラスタ）
   2. 壁にぶつかったらバックしてみよう（追加制御）
   3. PID制御で舵角値をいい感じにしよう（制御の改善）
   4. MLを試そう（ルールベースの代替）
4. 分析
   1. センサーの値を確認しよう（予想と実測値、バラツキ、不具合）
   2. 走行記録を視覚化してみよう（グラフ、画像、動画）

## ハードウェア
### 制限部門貸し出しマシン
| 分類 | 名称 | 個数 | 概算コスト(円) | 説明 |
| ---- | ---- | ---- | ---- | ---- |
| コンピュータ | [ラズパイ3B+](https://www.raspberrypi.com/products/raspberry-pi-3-model-b-plus/) |1| ---- |
| コンピュータ | [ラズパイ3A](https://raspberry-pi.ksyic.com/main/index/pdp.id/512/pdp.open/512) |-| 5000 |（代替）|
| コンピュータ | [ラズパイ4B](https://akizukidenshi.com/catalog/g/g114839/) |-| 10000 |（代替）|
| SDカード | 配布時期による |1| 64GB以上、書き込み速度30MB/s以上推奨 |
| 距離センサ | [超音波距離センサー HC-SR04](https://akizukidenshi.com/catalog/g/g111009/) |5| 1500 |
| カメラ | [ラズベリー•パイ（Raspberry Pi）160°広角500MP](https://jp.sainsmart.com/products/wide-angle-fov160-5-megapixel-camera-module-for-raspberry-pi) |任意| 3000 |
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
| togikai基盤 | --- |任意|  HC-SR04用の5V->3.3Vレベルシフタ等搭載 |


## ラズパイのセットアップ
1. OS:[2021-01-11-raspios-buster-i386.iso](https://downloads.raspberrypi.com/rpd_x86/images/rpd_x86-2021-01-12/2021-01-11-raspios-buster-i386.iso)
2. Raspberry [Pi Imager](https://www.raspberrypi.com/software/)を使ってSDカードへ書き込み
3. [togikaidrive](https://github.com/autonomous-minicar-battle/togikaidrive.git)をgit cloneする



## ツール類
- エディター：[VS](https://code.visualstudio.com/) 
- コード管理：[Git](https://git-scm.com/) 
- コード配布：[GitHub](https://github.com/) 
 - GUIでファイル転送：[Filezilla](https://filezilla-project.org/)
- []()

　