# coding:utf-8
import datetime
import os

# 判断モード選択
model_plan_list = ["GoStraight",
                   "Right_Left_3","Right_Left_3_Records",
                   "RightHand","RightHand_PID","LeftHand","LeftHand_PID",
                   "NN"]
mode_plan = "Right_Left_3_Records"
# 判断モード関連パラメータ
## 過去の操作値記録回数
motor_Nrecords = 3

# 復帰モード選択
mode_recovery = "Back" #None, Back, Stop
recovery_time = 0.1

# 出力系
# 判断結果出力、Thonyのplotterを使うならFalse
print_plan_result = False
# Thonnyのplotterを使う場合
plotter = False

# モーター出力パラメータ （デューティー比：-100~100で設定）
# スロットル用
FORWARD_S = 70 #ストレートでの値, joy_accel1
FORWARD_C = 55 #カーブでのの値, joy_accel2
STOP = 0
REVERSE = -100 
# ステアリング用
LEFT = 100 #<=100
NUTRAL = 0 
RIGHT = -100 #<=100

# 超音波センサの検知パラメータ 
## 距離関連、単位はmm
### 前壁の停止/検知距離
DETECTION_DISTANCE_STOP = 250
DETECTION_DISTANCE_BACK = 150
DETECTION_DISTANCE_Fr = 150
### 右左折判定基準距離
DETECTION_DISTANCE_RL = 550
### 他
DETECTION_DISTANCE_FrLH = 350
DETECTION_DISTANCE_FrRH = 350
DETECTION_DISTANCE_RrLH = 350
DETECTION_DISTANCE_RrRH = 350
DETECTION_DISTANCE_TARGET = 200 #目標距離
DETECTION_DISTANCE_RANGE = 50/2 #修正認知半径距離

## PIDパラメータ(PDまでを推奨)
K_P = 0.7 #0.7
K_I = 0.0 #0.0
K_D = 0.3 #0.3

#↑↑↑体験型イベント向けパラメータはここまで↑↑↑～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
# 車両調整用パラメータ(motor.pyで調整した後値を入れる)
## ステアリングのPWMの値
STEERING_CENTER_PWM = 410 #410:newcar, #340~360:oldcar
STEERING_WIDTH_PWM = 100
STEERING_RIGHT_PWM = STEERING_CENTER_PWM + STEERING_WIDTH_PWM
STEERING_LEFT_PWM = STEERING_CENTER_PWM - STEERING_WIDTH_PWM
### !!!ステアリングを壊さないための上限下限の値設定  
STEERING_RIGHT_PWM_LIMIT = 550
STEERING_LEFT_PWM_LIMIT = 250

## アクセルのPWM値
## モーターの回転音を聞き、音が変わらないところが最大/最小値とする
THROTTLE_STOPPED_PWM = 380 #390:newcar, #370~390:oldcar
THROTTLE_FORWARD_PWM = 500
THROTTLE_REVERSE_PWM = 250
THROTTLE_WIDTH_PWM = 100 

# 超音波センサの設定
## 使う超音波センサ位置の指示、計測ループが遅い場合は数を減らす
### 前３つ使う場合はこちらをコメントアウト外す
ultrasonics_list = ["FrLH","Fr","FrRH"]
### ５つ使う場合はこちらをコメントアウト外す
#ultrasonics_list = ["RrLH", "FrLH", "Fr", "FrRH","RrRH"]
### ８つ使う場合ははこちらのコメントアウト外す
#ultrasonics_list.extend(["BackRH", "Back", "BackLH"])
### ほかのファイルで使うためリスト接続名
ultrasonics_list_join = "uls_"+"_".join(ultrasonics_list)

## 超音波センサの測定回数、ultrasonic.pyチェック用
sampling_times = 100
## 目標サンプリング周期（何秒に１回）、複数センサ利用の場合は合計値、
sampling_cycle = 0.01
## 過去の超音波センサの値記録回数
ultrasonics_Nrecords = 3

# GPIOピン番号:超音波センサの位置の対応とPWMピンのチャンネル
## 新旧ボードの指定
board = "new" #old：~2023年たこ足配線、new：新ボード

## !!!超音波センサとPWMの配線を変えない限り触らない
if board == "old":
    ### Echo -- Fr:26, FrLH:24, RrLH:37, FrRH:31, RrRH:38
    e_list=[26,24,37,31,38]
    ### Triger -- Fr:15, FrLH:13, RrLH:35, FrRH:32, RrRH:36
    t_list=[15,13,35,32,36]
    ultrasonics_dict_trig = {"Fr":t_list[0], "FrLH":t_list[1], "RrLH":t_list[2], "FrRH":t_list[3], "RrRH":t_list[4]} 
    ultrasonics_dict_echo = {"Fr":e_list[0], "FrLH":e_list[1], "RrLH":e_list[2], "FrRH":e_list[3], "RrRH":e_list[4]} 
    CHANNEL_STEERING = 14 #old board
    CHANNEL_THROTTLE = 13 #old board

elif board == "new": #new board
    ### Echo -- Fr:26, FrLH:24, RrLH:37, FrRH:31, RrRH:38
    e_list=[11,13,15,29,31,33,35,37]
    ### Triger -- Fr:15, FrLH:13, RrLH:35, FrRH:32, RrRH:36
    t_list=[12,16,18,22,32,36,38,40]
    ultrasonics_dict_trig = {"Fr":t_list[0], "FrRH":t_list[1], "FrLH":t_list[2], "RrRH":t_list[3], "RrLH":t_list[4]} 
    ultrasonics_dict_echo = {"Fr":e_list[0], "FrRH":e_list[1], "FrLH":e_list[2], "RrRH":e_list[3], "RrLH":e_list[4]} 
    CHANNEL_STEERING = 1 #new board
    CHANNEL_THROTTLE = 0 #new board

else:
    print("Please set board as 'old' or 'new'.")

N_ultrasonics = len(ultrasonics_list)

# NNパラメータ
HAVE_NN = False
if mode_plan == "NN": HAVE_NN = True

## 学習済みモデルのパス
model_dir = "models"
model_name = "model_20240709_record_20240624_023159.csv_epoch_30_uls_RrLH_FrLH_Fr_FrRH_RrRH.pth"
model_path = os.path.join(model_dir, model_name)
## モデルと学習のハイパーパラメータ設定
hidden_dim = 64 #（隠れ層のノード数）
num_hidden_layers = 3 #（隠れ層の数）
batch_size = 8

## モデルの種類
model_type = "categorical" #linear, categorical
# カテゴリの設定、カテゴリ数は揃える↓　
num_categories = 3
# -100~100の範囲で小さな値→大きな値の順にする（しないとValueError: bins must increase monotonically.）
categories_Str = [RIGHT, NUTRAL, LEFT]
categories_Thr = [FORWARD_C, FORWARD_S, FORWARD_C] #Strに合わせて設定

bins_Str = [-101] # -101は最小値-100を含むため設定、境界の最大値は100
#bins_Thr = [-101]
# 分類の境界：binを設定(pd.cutで使う)
for i in range(num_categories):
    bins_Str.append((categories_Str[i]+categories_Str[min(i+1,num_categories-1)])/2)
bins_Str[-1] = 100
#for i in range(num_categories):
#    bins_Thr.append((categories_Thr[i]+categories_Thr[min(i+1,num_categories-1)])/2)
#bins_Thr[-1] = 100

# コントローラー（ジョイスティックの設定）
HAVE_CONTROLLER = True #True
JOYSTICK_STEERING_SCALE = -1.0       #some people want a steering that is less sensitve. This scalar is multiplied with the steering -1 to 1. It can be negative to reverse dir.
JOYSTICK_THROTTLE_SCALE = -1.0       #some people want a throttle that is less sensitve. 
#AUTO_RECORD_ON_THROTTLE = False      #if true, we will record whenever throttle is not zero. if false, you must manually toggle recording with some other trigger. Usually circle button on joystick.
#CONTROLLER_TYPE = 'F710'            #(ps3|ps4|xbox|pigpio_rc|nimbus|wiiu|F710|rc3|MM1|custom) custom will run the my_joystick.py controller written by the `donkey createjs` command
JOYSTICK_DEVICE_FILE = "/dev/input/js0" 
## ジョイスティックのボタンとスティック割り当て
# F710の操作設定 #割り当て済み
JOYSTICK_A = 0 #アクセル１
JOYSTICK_B = 1 #アクセル２
JOYSTICK_X = 2 #ブレーキ
JOYSTICK_Y = 3 #記録停止開始
JOYSTICK_LB = 4
JOYSTICK_RB = 5
JOYSTICK_BACK = 6
JOYSTICK_S = 7 #自動/手動走行切り替え
JOYSTICK_Logi = 8
JOYSTICK_LSTICKB = 9
JOYSTICK_RSTICKB = 10
JOYSTICK_AXIS_LEFT = 0 #ステアリング（左右）
JOYSTICK_AXIS_RIGHT = 4 #スロットル（上下）
JOYSTICK_HAT_LR = 0
JOYSTICK_HAT_DU = 1

# カメラの設定
HAVE_CAMERA = False
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
#CAMERA_FRAMERATE = 20 #DRIVE_LOOP_HZ
#CAMERA_VFLIP = False
#CAMERA_HFLIP = False
#IMSHOW = False #　画像を表示するか

#↑↑↑ルールベース/機械学習講座向けパラメータはここまで↑↑↑～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
# その他
# ジャイロを使った動的制御モード選択
HAVE_IMU = False #True
mode_dynamic_control = "GCounter" #"GCounter", "GVectoring"

# FPV 下記のport番号
## fpvがONの時は画像保存なし
fpv = False #True
port = 8910

# 走行記録
## 測定データ
records = "records"
if not os.path.exists(records):
    # ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs(records)
    print("make dir as ",records)
## 記録したcsvファイル名
record_filename = './'+records+'/record_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'

# 画像保存
img_size = (IMAGE_W, IMAGE_H, IMAGE_DEPTH)
images = "images"
if not os.path.exists(images):
    # ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs(images)
    print("make dir as ",images)
## 記録するフォルダ名
image_dir = './'+images+'/image_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(image_dir)
print("make dir as ",image_dir)
