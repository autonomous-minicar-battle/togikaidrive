# coding:utf-8
import config

# ~~~出前授業用に一部のバラメータを変更
# ！！！出前授業用に生徒が変更するバラメータ　ここから ！！！　#
# モーター出力パラメータ （デューティー比：-100~100で設定）
# スロットル用
config.FORWARD_S = 80 #ストレートでの値, joy_accel1
config.FORWARD_C = 50 #カーブでのの値, joy_accel2
config.REVERSE = -40 
# ステアリング用
config.LEFT = 100 #<=100
config.RIGHT = -100 #<=100

# 超音波センサの検知パラメータ 
## 距離関連、単位はmm
### 前壁の停止/検知距離
config.DETECTION_DISTANCE_STOP = 150
config.DETECTION_DISTANCE_BACK = 150
config.DETECTION_DISTANCE_Fr = 500
### 右左折判定基準距離
config.DETECTION_DISTANCE_RL = 700
### 他
config.DETECTION_DISTANCE_FrLH = 150
config.DETECTION_DISTANCE_FrRH = 150
config.DETECTION_DISTANCE_RrLH = 150
config.DETECTION_DISTANCE_RrRH = 150
config.DETECTION_DISTANCE_TARGET = 180 #目標距離
config.DETECTION_DISTANCE_RANGE = 60/2 #修正認知半径距離

# 判断モード選択
##　選択肢："Right_Left_3","Right_Left_3_Records","RightHand","RightHand_PID"
config.mode_plan = "Right_Left_3_Records" #"GoStraight"

# ！！！出前授業用に生徒が変更するバラメータ　ここまで ！！！　#

# 復帰モード選択
config.mode_recovery = "Back" #None, Back, Stop
config.recovery_time = 0.5

# 超音波センサー設定
### 前３つ使う場合はこちらをコメントアウト外す
#config.ultrasonics_list = ["FrLH","Fr","FrRH"]
### ５つ使う場合はこちらをコメントアウト外す
config.ultrasonics_list = ["RrLH", "FrLH", "Fr", "FrRH","RrRH"]


### 新旧ボードの選択
config.board = "new" #old：~2023年たこ足配線、new：新ボード
### GPIOピン番号の指示方法と超音波センサの位置の対応とPWMピンのチャンネル
## !!!超音波センサとPWMの配線を変えない限り触らない
if config.board == "old":
    ### Echo -- Fr:26, FrLH:24, RrLH:37, FrRH:31, RrRH:38
    config.e_list=[26,24,37,31,38]
    ### Triger -- Fr:15, FrLH:13, RrLH:35, FrRH:32, RrRH:36
    config.t_list=[15,13,35,32,36]
    config.ultrasonics_dict_trig = {"Fr":config.t_list[0], "FrLH":config.t_list[1], "RrLH":config.t_list[2], "FrRH":config.t_list[3], "RrRH":config.t_list[4]} 
    config.ultrasonics_dict_echo = {"Fr":config.e_list[0], "FrLH":config.e_list[1], "RrLH":config.e_list[2], "FrRH":config.e_list[3], "RrRH":config.e_list[4]} 
    config.CHANNEL_STEERING = 14
    config.CHANNEL_THROTTLE = 13

elif config.board == "new": #new board
    ### Echo -- Fr:26, FrLH:24, RrLH:37, FrRH:31, RrRH:38
    config.e_list=[11,13,15,29,31,33,35,37]
    ### Triger -- Fr:15, FrLH:13, RrLH:35, FrRH:32, RrRH:36
    config.t_list=[12,16,18,22,32,36,38,40]
    config.ultrasonics_dict_trig = {"Fr":config.t_list[0], "FrRH":config.t_list[1], "FrLH":config.t_list[2], "RrRH":config.t_list[3], "RrLH":config.t_list[4]}
    config.ultrasonics_dict_echo = {"Fr":config.e_list[0], "FrRH":config.e_list[1], "FrLH":config.e_list[2], "RrRH":config.e_list[3], "RrLH":config.e_list[4]}
    config.CHANNEL_STEERING = 1
    config.CHANNEL_THROTTLE = 0
else:
    print("Please set board as 'old' or 'new'.")

## 操舵のPWM値
config.STEERING_CENTER_PWM = 390
config.STEERING_WIDTH_PWM = 80
config.STEERING_RIGHT_PWM = config.STEERING_CENTER_PWM + config.STEERING_WIDTH_PWM
config.STEERING_LEFT_PWM = config.STEERING_CENTER_PWM - config.STEERING_WIDTH_PWM
### !!!ステアリングを壊さないための上限下限の値設定  
config.STEERING_RIGHT_PWM_LIMIT = 550
config.STEERING_LEFT_PWM_LIMIT = 250

## アクセルのPWM値(motor.pyで調整した後値を入れる)
## モーターの回転音を聞き、音が変わらないところが最大/最小値とする
config.THROTTLE_STOPPED_PWM = 370
config.THROTTLE_FORWARD_PWM = 500
config.THROTTLE_REVERSE_PWM = 240

## thonny plotterで値を見るときにTrue
config.plotter = False
## 使わないライブラリOFF
config.HAVE_CAMERA =False
config.HAVE_IMU =False
config.HAVE_CONTROLLER = True
config.fpv = False
# ~~~出前授業用に一部のバラメータを変更　ここまで ~~~


# 一般的な外部ライブラリ
import os
import RPi.GPIO as GPIO
GPIO.setwarnings(False)
## GPIOピン番号の指示方法
GPIO.setmode(GPIO.BOARD)
GPIO.setup(config.e_list,GPIO.IN)
GPIO.setup(config.t_list,GPIO.OUT,initial=GPIO.LOW)

import time
import numpy as np
import sys
import multiprocessing
from multiprocessing import Process

print("ライブラリの初期化に数秒かかります...")
# togikaidriveのモジュール
import ultrasonic
import motor
import planner
# 以下はconfig.pyでの設定によりimport
if config.HAVE_CONTROLLER: import joystick
if config.HAVE_CAMERA: import camera_multiprocess
if config.HAVE_IMU: import gyro
if config.HAVE_NN: import train_pytorch

# First Person Viewでの走行画像表示
if config.fpv:
    #img_sh = multiprocessing.sharedctypes.RawArray('i', config.img_size[0]*config.img_size[1]*config.img_size[2])
    data_sh = multiprocessing.sharedctypes.RawArray('i', (2,3))
    import fpv
    server = Process(target = fpv.run,  args = data_sh, kwargs = {'host': 'localhost', 'port': config.port, 'threaded': True})
    server.start()
   #fpv.run(host='localhost', port=config.port, debug=False, threaded=True)

# データ記録用配列作成
d = np.zeros(config.N_ultrasonics)
d_stack = np.zeros(config.N_ultrasonics+3)
recording = True

# 画像保存
#running = Value("b", True)
if config.HAVE_CAMERA and not config.fpv:
    print("Start taking pictures in ",config.image_dir)
    cam = camera_multiprocess.VideoCaptureWrapper(0)
    print("【 ◎*】Capture started! \n")
    #cam.__buffer


# 操舵、駆動モーターの初期化
motor = motor.Motor()
motor.set_throttle_pwm_duty(config.STOP)
motor.set_steer_pwm_duty(config.NUTRAL)

# 超音波センサの初期化
## 別々にインスタンス化する例　ultrasonic_RrLH = ultrasonic.Ultrasonic("RrLH")
## 一気にnameに"RrLH"等をultrasonics_listから入れてインスタンス化
ultrasonics = {name: ultrasonic.Ultrasonic(name=name) for name in config.ultrasonics_list}
print(" 下記の超音波センサを利用")
print(" ", config.ultrasonics_list)

# 操作判断プランナーの初期化
plan = planner.Planner(config.mode_plan)

# NNモデルの読み込み
if config.HAVE_NN:
    ## NNモデルの初期化
    ## 使う超音波センサの数、出力数、隠れ層の次元、隠れ層の数
    model = train_pytorch.NeuralNetwork(
        len(config.ultrasonics_list), 2,
        config.hidden_dim, config.num_hidden_layers)
    ## 保存したモデルをロード
    print("\n保存したモデルをロードします: ", config.model_path)
    train_pytorch.load_model(model, config.model_path, None, config.model_dir)
    print(model)

#　imuの初期化
if config.HAVE_IMU:
    imu = gyro.BNO055()
    ## 計測例
    ## angle, acc, gyr = imu.measure_set()

# コントローラーの初期化
mode = "auto"
if config.HAVE_CONTROLLER:
    joystick = joystick.Joystick()

# 一時停止（Enterを押すとプログラム実行開始）
print('Enterを押して走行開始!')
input()

# 途中でモータースイッチを切り替えたとき用に再度モーター初期化
# 初期化に成功するとピッピッピ！と３回音がなる、失敗時（PWMの値で約370-390以外の値が入りっぱなし）はピ...ピ...∞
motor.set_throttle_pwm_duty(config.STOP)

# fpv
## pass

# 開始時間
start_time = time.time()

# ここから走行ループ
try:
    while True:
        # 認知（計測） ＃
        ## RrRHセンサ距離計測例：dis_RrRH = ultrasonic_RrRH.()
        ## 下記では一気に取得
        message = ""
        for i, name in enumerate(config.ultrasonics_list):
            d[i] = ultrasonics[name].measure()
            #message += name + ":" + str(round(ultrasonics[name].dis,2)).rjust(7, ' ') #Thony表示用にprint変更
            message += name + ":" + "{:>4}".format(round(ultrasonics[name].dis))+ ", "
            # サンプリングレートを調整する場合は下記をコメントアウト外す
            #time.sleep(sampling_cycle)

        # 判断（プランニング）＃
        # 使う超音波センサをconfig.pyのultrasonics_listで設定必要
        ## ただ真っすぐに走る 
        if config.mode_plan == "GoStraight":
            steer_pwm_duty,throttle_pwm_duty = 0, config.FORWARD_S
        ## 右左空いているほうに走る 
        elif config.mode_plan == "Right_Left_3":
            steer_pwm_duty,throttle_pwm_duty = plan.Right_Left_3(ultrasonics["FrLH"].dis, ultrasonics["Fr"].dis, ultrasonics["FrRH"].dis)
        ## 過去の値を使ってスムーズに走る
        elif config.mode_plan == "Right_Left_3_Records":
            steer_pwm_duty, throttle_pwm_duty  = plan.Right_Left_3_Records(ultrasonics["FrLH"].dis, ultrasonics["Fr"].dis, ultrasonics["FrRH"].dis)
        ## 右手法で走る
        elif config.mode_plan == "RightHand":
            steer_pwm_duty, throttle_pwm_duty  = plan.RightHand(ultrasonics["FrRH"].dis, ultrasonics["RrRH"].dis)
        ## 右手法にPID制御を使ってスムーズに走る
        elif config.mode_plan == "RightHand_PID":
            steer_pwm_duty, throttle_pwm_duty  = plan.RightHand_PID(ultrasonics["FrRH"], ultrasonics["RrRH"])
        ## ニューラルネットを使ってスムーズに走る
        #評価中
        elif config.mode_plan == "NN":
            # 超音波センサ入力が変更できるように引数をリストにして渡す形に変更
            args = [ultrasonics[key].dis for key in config.ultrasonics_list]
            steer_pwm_duty, throttle_pwm_duty = plan.NN(model, *args)
            #steer_pwm_duty, throttle_pwm_duty  = plan.NN(model, ultrasonics["FrLH"].dis, ultrasonics["Fr"].dis, ultrasonics["FrRH"].dis)
        else: 
            print("デフォルトの判断モードの選択ではありません, コードを書き換えてオリジナルのモードを実装しよう!")
            break

        # 操作（ステアリング、アクセル）＃
        ## ジョイスティックで操作する場合は上書き
        if config.HAVE_CONTROLLER:
            joystick.poll()
            mode = joystick.mode[0]
            if mode == "user":
                steer_pwm_duty = int(joystick.steer*config.JOYSTICK_STEERING_SCALE*100)
                throttle_pwm_duty = int(joystick.accel*config.JOYSTICK_THROTTLE_SCALE*100)
                if joystick.accel2:
                    throttle_pwm_duty  = int(config.FORWARD_S)
                elif joystick.accel1:
                    throttle_pwm_duty  = int(config.FORWARD_C)
            if joystick.recording: 
                recording = True
            else: 
                recording = False
            
            ### コントローラでブレーキ
            if joystick.breaking:
                motor.breaking()

        ## モータードライバーに出力をセット
        ### 補正（動的制御）
        ### Gthr:スロットル（前後方向）のゲイン、Gstr:ステアリング（横方向）のゲイン
        ### ヨー角の角速度でオーバーステア/スリップに対しカウンターステア 
        if config.mode_plan == "GCounter":
            imu.GCounter()
            motor.set_steer_pwm_duty(steer_pwm_duty * (1 - 2 * imu.Gstr))        
            motor.set_throttle_pwm_duty(throttle_pwm_duty * (1 - 2 * imu.Gthr))
        ## ヨー角の角速度でスロットル調整 
        ## 未実装
        #elif config.mode_plan == "GVectoring":
        #    imu.GVectoring()
        else: 
            motor.set_steer_pwm_duty(steer_pwm_duty)        
            motor.set_throttle_pwm_duty(throttle_pwm_duty)  

        ## 記録（タイムスタンプと距離データを配列に記録）
        ts =  time.time()
        ts_run =  round(ts-start_time,2)
        if recording:
            d_stack = np.vstack((d_stack, np.insert(d, 0, [ts, steer_pwm_duty, throttle_pwm_duty]),))
            ### 画像保存 ret:カメラ認識、img：画像
            if config.HAVE_CAMERA and not config.fpv:
                ret, img = cam.read()
                cam.save(img, ts, steer_pwm_duty, throttle_pwm_duty, config.image_dir)

        ## 全体の状態を出力      
        #print("Rec:"+recording, "Mode:",mode,"RunTime:",ts_run ,"Str:",steer_pwm_duty,"Thr:",throttle_pwm_duty,"Uls:", message) #,end=' , '
        if mode == 'auto' : mode = config.mode_plan
        if config.plotter:
              print(message)
        else: print("Rec:{0}, Mode:{1}, RunTime:{2:>5}, Str:{3:>4}, Thr:{4:>4}, Uls:[ {5}]".format(recording, mode, ts_run, steer_pwm_duty, throttle_pwm_duty, message)) #,end=' , '

        ## 後退/停止操作（簡便のため、判断も同時に実施） 
        if config.mode_recovery == "None":
            pass

        elif config.mode_recovery == "Back":  
            ### 後退
            plan.Back(ultrasonics["Fr"])
            if plan.flag_back == True:
                motor.set_steer_pwm_duty(config.NUTRAL)
                motor.set_throttle_pwm_duty(config.REVERSE)
                time.sleep(config.recovery_time)
            else: 
                pass

        elif config.mode_recovery == "Stop":
            ### 停止
            plan.Stop(ultrasonics["Fr"])
            if plan.flag_stop ==True:
                ## 停止動作
                motor.set_steer_pwm_duty(config.NUTRAL)
                motor.set_throttle_pwm_duty(config.STOP)
                time.sleep(0.05)
                motor.set_throttle_pwm_duty(config.REVERSE)
                time.sleep(0.05)
                motor.set_throttle_pwm_duty(config.STOP)
                print("一時停止、Enterを押して走行再開!")
                input()
                #break

finally:
    # 終了処理
    print('\n停止')
    motor.set_throttle_pwm_duty(config.STOP)
    motor.set_steer_pwm_duty(config.NUTRAL)
    GPIO.cleanup()
    header ="Tstamp,Str,Thr,"
    for name in config.ultrasonics_list:
        header += name + ","
    header = header[:-1]        
    np.savetxt(config.record_filename, d_stack[1:], delimiter=',',  fmt='%10.2f', header=header, comments="")
    #np.savetxt(config.record_filename, d_stack[1:], fmt='4f',header=header, comments="")
    print('記録停止')
    print("記録保存--> ",config.record_filename)
    print("画像保存--> ",config.image_dir)

header ="Tstamp, Str, Thr, "
for name in config.ultrasonics_list:
    header += name + ", "
header = header[:-1]        