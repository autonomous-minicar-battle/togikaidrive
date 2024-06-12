# coding:utf-8
# 一般的な外部ライブラリ
import os
import RPi.GPIO as GPIO
import time
import numpy as np
import sys
import multiprocessing
from multiprocessing import Process

print("ライブラリの初期化に数秒かかります...")
# togikaidriveのモジュール
#import config
import ultrasonic
import motor
import planner
import joystick
#import camera_multiprocess
import gyro
#import train_pytorch

class Config:
    def __init__(self):
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        import datetime
        import os

        # モーター出力パラメータ （デューティー比：-100~100で設定）
        # スロットル用
        self.FORWARD_S = 80 #ストレートでの値, joy_accel1
        self.FORWARD_C = 60 #カーブでのの値, joy_accel2
        self.STOP = 0
        self.REVERSE = -60 
        # ステアリング用
        self.LEFT = 100 #<=100
        self.NUTRAL = 0 
        self.RIGHT = -100 #<=100

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
        DETECTION_DISTANCE_FrLH = 150
        DETECTION_DISTANCE_FrRH = 150
        DETECTION_DISTANCE_TARGET = 180 #目標距離
        DETECTION_DISTANCE_RANGE = 60/2 #修正認知半径距離

        # 判断モード選択
        ##　選択肢："Right_Left_3","Right_Left_3_Records","RightHand","RightHand_PID"
        self.mode_plan = "Right_Left_3"
        model_plan_list = ["Right_Left_3","Right_Left_3_Records","RightHand","RightHand_PID"]
        ## 判断結果出力、Thonyのplotterを使うならFalse
        print_plan_result = False
        ## PIDパラメータ(PDまでを推奨)
        K_P = 0.7 #0.7
        K_I = 0.0 #0.0
        K_D = 0.3 #0.3

        # 復帰モード選択
        self.mode_recovery = "Back" #None, Back, Stop

        # 動的制御モード選択
        mode_dynamic_control = "GCounter" #GVectoring
        ## PWMピンのチャンネル
        ## !!!配線を変えない限り触らない
        CHANNEL_STEERING = 14
        CHANNEL_THROTTLE = 13
        #CHANNEL_STEERING = 1 #new board
        #CHANNEL_THROTTLE = 0 #new board

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

        #↑↑↑体験型イベント向けパラメータはここまで↑↑↑～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

        # 超音波センサ
        ## 使う超音波センサ位置の指示、計測ループが遅い場合は数を減らす
        ### 前３つ使う場合はこちらをコメントアウト外す
        self.ultrasonics_list = ["FrLH","Fr","FrRH"]
        ### ５つ使う場合はこちらをコメントアウト外す
        #ultrasonics_list = ["RrLH", "FrLH", "Fr", "FrRH","RrRH"]
        ### ８つ使う場合ははこちらのコメントアウト外す
        #ultrasonics_list.extend(["BackRH", "Back", "BackLH"])

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
        e_list=[26,24,37,31,38]
        #e_list=[11,13,15,29,31,33,35,37] #new board
        GPIO.setup(e_list,GPIO.IN)
        ### Triger -- Fr:15, FrLH:13, RrLH:35, FrRH:32, RrRH:36
        t_list=[15,13,35,32,36]
        #t_list=[12,16,18,22,32,36,38,40] #new board 
        GPIO.setup(t_list,GPIO.OUT,initial=GPIO.LOW)

        ## !!!超音波センサ初期設定、配線を変えない限り触らない
        ultrasonics_dict_trig = {"Fr":t_list[0], "FrRH":t_list[1], "FrLH":t_list[2], "RrRH":t_list[3], "RrLH":t_list[4], "BackRH":t_list[5], "Back":t_list[6], "BackLH":t_list[7]} 
        ultrasonics_dict_echo = {"Fr":e_list[0], "FrRH":e_list[1], "FrLH":e_list[2], "RrRH":e_list[3], "RrLH":e_list[4], "BackRH":e_list[5], "Back":e_list[6], "BackLH":e_list[7]}
        self.N_ultrasonics = len(self.ultrasonics_list)
        ## !!!

        # スロットル/ステアリングモーター用 パラメーター
        ## 過去の操作値記録回数
        motor_Nrecords = 5

        # 走行記録
        ## 測定データ
        records = "records"
        if not os.path.exists(records):
            # ディレクトリが存在しない場合、ディレクトリを作成する
            os.makedirs(records)
            print("make dir as ",records)
        ## 記録したcsvファイル名
        self.record_filename = './'+records+'/record_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'

config = Config()

# データ記録用配列作成
d = np.zeros(config.N_ultrasonics)
d_stack = np.zeros(config.N_ultrasonics+3)
recording = True


# 操舵、駆動モーターの初期化
motor = motor.Motor()
motor.set_throttle_pwm_duty(config.STOP)
motor.set_steer_pwm_duty(config.NUTRAL)

# 超音波センサの初期化
## 別々にインスタンス化する例　ultrasonic_RrLH = ultrasonic.Ultrasonic("RrLH")
## 一気にnameに"RrLH"等をultrasonics_listから入れてインスタンス化
ultrasonics = {name: ultrasonic.Ultrasonic(name=name) for name in config.ultrasonics_list}
print(" 下記の", config.N_ultrasonics,"個の超音波センサを利用")
print(" ", config.ultrasonics_list)


# 操作判断プランナーの初期化
plan = planner.Planner(config.mode_plan)
mode = "auto"

# 一時停止（Enterを押すとプログラム実行開始）
print('Enterを押して走行開始!')
input()

# 途中でモータースイッチを切り替えたとき用に再度モーター初期化
# 初期化に成功するとピッピッピ！と３回音がなる、失敗時（PWMの値で約370-390以外の値が入りっぱなし）はピ...ピ...∞
motor.set_throttle_pwm_duty(config.STOP)

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
        ## 右左空いているほうに走る 
        if config.mode_plan == "Right_Left_3":
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
            #steer_pwm_duty, throttle_pwm_duty  = plan.NN(model, ultrasonics["FrLH"].dis, ultrasonics["Fr"].dis, ultrasonics["FrRH"].dis)
        else: 
            print("デフォルトの判断モードの選択ではありません, コードを書き換えてオリジナルのモードを実装しよう!")
            break

        # 操作（ステアリング、アクセル）＃
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

        ## 全体の状態を出力      
        #print("Rec:"+recording, "Mode:",mode,"RunTime:",ts_run ,"Str:",steer_pwm_duty,"Thr:",throttle_pwm_duty,"Uls:", message) #,end=' , '
        if mode == 'auto' : mode = config.mode_plan
        print("Rec:{0}, Mode:{1}, RunTime:{2:>5}, Str:{3:>4}, Thr:{4:>4}, Uls:[ {5}]".format(recording, mode, ts_run, steer_pwm_duty, throttle_pwm_duty, message)) #,end=' , '

        ## 後退/停止操作（簡便のため、判断も同時に実施） 
        if config.mode_recovery == "None":
            pass

        elif config.mode_recovery == "Back":  
            ### 後退
            plan.Back(ultrasonics["Fr"])
            if plan.flag_back == True:
                motor.set_steer_pwm_duty(config.NUTRAL)
                motor.set_throttle_pwm_duty(config.REVERSE)
                time.sleep(0.9)
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
    header ="Tstamp, Str, Thr, "
    for name in config.ultrasonics_list:
        header += name + ", "        
    np.savetxt(config.record_filename, d_stack[1:], delimiter=',',  fmt='%10.2f', header=header, comments="")
    #np.savetxt(config.record_filename, d_stack, fmt='%.3e',header=header, comments="")
    print('記録停止')
    print("記録保存--> ",config.record_filename)

