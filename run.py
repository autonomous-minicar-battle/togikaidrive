# coding:utf-8
import os
import RPi.GPIO as GPIO
import time
import numpy as np
import sys

print("ライブラリの初期化に数秒かかります...")
import config
import ultrasonic
import motor
import planner
import joystick
import camera_multiprocess
import multiprocessing
from multiprocessing import Process
import gyro

if config.fpv:
    #img_sh = multiprocessing.sharedctypes.RawArray('i', config.img_size[0]*config.img_size[1]*config.img_size[2])
    data_sh = multiprocessing.sharedctypes.RawArray('i', (2,3))
    import fpv
    #server = Process(target = fpv.run,  kwargs = {'host': 'localhost', 'port': config.port, 'threaded': True})
    server = Process(target = fpv.run,  args = data_sh, kwargs = {'host': 'localhost', 'port': config.port, 'threaded': True})
    #server = Process(target = fpv.run, args = img_sh, kwargs = {'host': 'localhost', 'port': config.port, 'threaded': True})
    server.start()
   #fpv.run(host='localhost', port=config.port, debug=False, threaded=True)

#while True:
#    print (fpv.frame)
    #pass

# データ記録用配列作成
d = np.zeros(config.N_ultrasonics)
d_stack = np.zeros(config.N_ultrasonics+3)
recording = True

# 画像保存
#running = Value("b", True)
if not config.fpv:
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
print(" 下記の", config.N_ultrasonics,"個の超音波センサを利用")
print(" ", config.ultrasonics_list)

#　imu インスタンス化
imu = gyro.BNO055()
## 計測例
## angle, acc, gyr = imu.measure_set()

# 操作判断プランナーの初期化
plan = planner.Planner(config.mode_plan)

# コントローラーの初期化
if config.CONTROLLER:
    joystick = joystick.Joystick()

# 一時停止（Enterを押すとプログラム実行開始）
print('Enterを押して走行開始!')
input()

# 途中でモータースイッチを切り替えたとき用に再度モーター初期化
# 初期化に成功するとピッピッピ！と３回音がなる、失敗時（PWMの値で約370-390以外の値が入りっぱなし）はピ...ピ...∞
motor.set_throttle_pwm_duty(config.STOP)

# fpv

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
            message += name + ":" + str(round(ultrasonics[name].dis,2))+ ", "
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
        #未実装
        #steer_pwm_duty, throttle_pwm_duty  = plan.NN(dis_FrRH, dis_RrRH)
        else: 
            print("デフォルトの判断モードの選択ではありません, コードを書き換えてオリジナルのモードを実装しよう!")
            break

        # 操作（ステアリング、アクセル）＃
        ## ジョイスティックで操作する場合は上書き
        if config.CONTROLLER:
            joystick.poll()
            if joystick.mode[0] == "user":
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

        ## 補正（動的制御）
        ## Gthr:スロットル（前後方向）のゲイン、Gstr:ステアリング（横方向）のゲイン
        ## ヨー角の角速度でオーバーステア/スリップに対しカウンターステア 
        if config.mode_plan == "GCounter":
            imu.GCounter()
        ## ヨー角の角速度でスロットル調整 
        ## 未実装
        #elif config.mode_plan == "GVectoring":
        #    imu.GVectoring()
        else: 
            pass

        ## モータードライバーに出力をセット
        motor.set_throttle_pwm_duty(throttle_pwm_duty * (1 - 2 * imu.Gthr))
        motor.set_steer_pwm_duty(steer_pwm_duty * (1 - 2 * imu.Gstr))        
        #motor.set_throttle_pwm_duty(throttle_pwm_duty)  
        #motor.set_steer_pwm_duty(steer_pwm_duty)        

        ## ブレーキ
        if joystick.breaking:
            motor.breaking()

        ## 記録（タイムスタンプと距離データを配列に記録）
        ts =  time.time()
        ts_run =  round(ts-start_time,2)
        if recording:
            d_stack = np.vstack((d_stack, np.insert(d, 0, [ts, steer_pwm_duty, throttle_pwm_duty]),))
            ### 画像保存 ret:カメラ認識、img：画像
            if not config.fpv:
                ret, img = cam.read()
                cam.save(img, ts, steer_pwm_duty, throttle_pwm_duty, config.image_dir)

        ## 全体の状態を出力      
        print("*Rec:",recording, "*Mode:",joystick.mode[0],"*RunTime:",ts_run ,"*Str:",steer_pwm_duty,"*Thr:",throttle_pwm_duty," ", message) #,end=' , '

        ## 後退/停止操作（簡便のため、判断も同時に実施） 
        if config.mode_recovery == "None":
            pass

        elif config.mode_recovery == "Back":  
            ### 後退
            plan.Back(ultrasonics["Fr"])
            if plan.flag_back == True:
                motor.set_throttle_pwm_duty(config.REVERSE)
                motor.set_steer_pwm_duty(config.NUTRAL)
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

    # 終了処理
    config.GPIO.cleanup()
    header ="Tstamp Str Thr "
    for name in config.ultrasonics_list:
        header += name + " "        
    np.savetxt(config.record_filename, d_stack[1:],  fmt='%10.2f', header=header, comments="")
    #np.savetxt(config.record_filename, d_stack, fmt='%.3e',header=header, comments="")
    print('記録停止')
    print("記録保存--> ",config.record_filename)
    print("画像保存--> ",config.image_dir)

# Ctr-C時の終了処理
except KeyboardInterrupt:
    motor.set_throttle_pwm_duty(config.STOP)
    motor.set_steer_pwm_duty(config.NUTRAL)
    print('\nユーザーにより停止')
    config.GPIO.cleanup()
    header ="Time Thr Str"
    for name in config.ultrasonics_list:
        header += name + " "        
    np.savetxt(config.record_filename, d_stack[1:],  fmt='%10.2f', header=header, comments="")
    #np.savetxt(config.record_filename, d_stack, fmt='%.3e',header=header, comments="")
    print('記録停止')
    print("記録保存--> ",config.record_filename)
    print("画像保存--> ",config.image_dir)
