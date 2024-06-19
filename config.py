# coding:utf-8
import RPi.GPIO as GPIO
GPIO.setwarnings(False)
import datetime
import os

# モーター出力パラメータ （デューティー比：-100~100で設定）
# スロットル用
FORWARD_S = 80 #ストレートでの値, joy_accel1
FORWARD_C = 60 #カーブでのの値, joy_accel2
STOP = 0
REVERSE = -60 
# ステアリング用
LEFT = 100 #<=100
NUTRAL = 0 
RIGHT = -100 #<=100

# 超音波センサの検知パラメータ 
## 距離関連、単位はmm
### 前壁の停止/検知距離
DETECTION_DISTANCE_STOP = 80
DETECTION_DISTANCE_BACK = 80
DETECTION_DISTANCE_Fr = 150
### 右左折判定基準距離
DETECTION_DISTANCE_RL = 150
### 他
DETECTION_DISTANCE_FrLH = 150
DETECTION_DISTANCE_FrRH = 150
DETECTION_DISTANCE_RrLH = 150
DETECTION_DISTANCE_RrRH = 150
DETECTION_DISTANCE_TARGET = 180 #目標距離
DETECTION_DISTANCE_RANGE = 60/2 #修正認知半径距離

# 判断モード選択
##　選択肢："Right_Left_3","Right_Left_3_Records","RightHand","RightHand_PID"
mode_plan = "Right_Left_3"
model_plan_list = ["GoStraight","Right_Left_3","Right_Left_3_Records","RightHand","RightHand_PID"]
## 判断結果出力、Thonyのplotterを使うならFalse
print_plan_result = False
## PIDパラメータ(PDまでを推奨)
K_P = 0.7 #0.7
K_I = 0.0 #0.0
K_D = 0.3 #0.3

# 復帰モード選択
mode_recovery = "Back" #None, Back, Stop
recovery_time = 0.5

# ジャイロを使った動的制御モード選択
HAVE_IMU = False #True
mode_dynamic_control = "GCounter" #GVectoring

# Thonnyのplotterを使う場合
plotter = False

#↑↑↑体験型イベント向けパラメータはここまで↑↑↑～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
# コントローラー（ジョイスティックの設定）
HAVE_CONTROLLER = False #True
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
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
CAMERA_FRAMERATE = 20 #DRIVE_LOOP_HZ
CAMERA_VFLIP = False
CAMERA_HFLIP = False
IMSHOW = False #　画像を表示するか


# NNパラメータ
#Nnode = 3
#Nlayer = 3
#model = "linear" #"categorical"
#Ncategory = 5

# 超音波センサ数
## 使う超音波センサ位置の指示、計測ループが遅い場合は数を減らす
### 前３つ使う場合はこちらをコメントアウト外す
#ultrasonics_list = ["FrLH","Fr","FrRH"]
### ５つ使う場合はこちらをコメントアウト外す
ultrasonics_list = ["RrLH", "FrLH", "Fr", "FrRH","RrRH"]
### ８つ使う場合ははこちらのコメントアウト外す
ultrasonics_list.extend(["BackRH", "Back", "BackLH"])

## 超音波センサの測定回数、ultrasonic.pyチェック用
sampling_times = 100
## 目標サンプリング周期（何秒に１回）、複数センサ利用の場合は合計値、
sampling_cycle = 0.05

## 過去の超音波センサの値記録回数
ultrasonics_Nrecords = 5

## 超音波センサ初期設定(配線を変えない限り触らない！)
## !!!超音波センサ初期設定、配線を変えない限り触らない
### GPIOピン番号の指示方法
GPIO.setmode(GPIO.BOARD)

### Echo -- Fr:26, FrLH:24, RrLH:37, FrRH:31, RrRH:38
#e_list=[26,24,37,31,38]
e_list=[11,13,15,29,31,33,35,37] #new board
GPIO.setup(e_list,GPIO.IN)
### Triger -- Fr:15, FrLH:13, RrLH:35, FrRH:32, RrRH:36
#t_list=[15,13,35,32,36]
t_list=[12,16,18,22,32,36,38,40] #new board 
GPIO.setup(t_list,GPIO.OUT,initial=GPIO.LOW)

## !!!超音波センサ初期設定、配線を変えない限り触らない
ultrasonics_dict_trig = {"Fr":t_list[0], "FrRH":t_list[1], "FrLH":t_list[2], "RrRH":t_list[3], "RrLH":t_list[4], "BackRH":t_list[5], "Back":t_list[6], "BackLH":t_list[7]} 
ultrasonics_dict_echo = {"Fr":e_list[0], "FrRH":e_list[1], "FrLH":e_list[2], "RrRH":e_list[3], "RrLH":e_list[4], "BackRH":e_list[5], "Back":e_list[6], "BackLH":e_list[7]}
#ultrasonics_dict_trig = {"Fr":t_list[0], "FrRH":t_list[1], "FrLH":t_list[2], "RrRH":t_list[3], "RrLH":t_list[4]} 
#ultrasonics_dict_echo = {"Fr":e_list[0], "FrRH":e_list[1], "FrLH":e_list[2], "RrRH":e_list[3], "RrLH":e_list[4]} 
N_ultrasonics = len(ultrasonics_list)
## !!!

# スロットル/ステアリングモーター用 パラメーター
## 過去の操作値記録回数
motor_Nrecords = 5

## PWMピンのチャンネル 配線を変えない限り触らない
#CHANNEL_STEERING = 14
#CHANNEL_THROTTLE = 13
CHANNEL_STEERING = 1 #new board
CHANNEL_THROTTLE = 0 #new board

## 操舵のPWM値
STEERING_CENTER_PWM = 360
STEERING_WIDTH_PWM = 80
STEERING_RIGHT_PWM = STEERING_CENTER_PWM + STEERING_WIDTH_PWM
STEERING_LEFT_PWM = STEERING_CENTER_PWM - STEERING_WIDTH_PWM
### !!!ステアリングを壊さないための上限下限の値設定  
STEERING_RIGHT_PWM_LIMIT = 450
STEERING_LEFT_PWM_LIMIT = 250


## アクセルのPWM値(motor.pyで調整した後値を入れる)
## モーターの回転音を聞き、音が変わらないところが最大/最小値とする
THROTTLE_STOPPED_PWM = 390
THROTTLE_FORWARD_PWM = 540
THROTTLE_REVERSE_PWM = 320
### 設定不要
THROTTLE_WIDTH_PWM = 80

# 走行記録
## 測定データ
records = "records"
if not os.path.exists(records):
    # ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs(records)
    print("make dir as ",records)
## 記録したcsvファイル名
record_filename = './'+records+'/record_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'

## 画像
HAVE_CAMERA = False
img_size = (120, 160, 3)
images = "images"
if not os.path.exists(images):
    # ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs(images)
    print("make dir as ",images)
## 記録するフォルダ名
image_dir = './'+images+'/image_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(image_dir)
print("make dir as ",image_dir)

# FPV 下記のport番号
## fpvがONの時は画像保存なし
fpv = False #True
port = 8910
