# coding:utf-8
import os
import RPi.GPIO as GPIO
import time
import numpy as np

import config
import ultrasonic
import motor
import planner
import joystick
import multiprocessing


# データ記録用配列作成
d = np.zeros(config.N_ultrasonics)
print(d)
d_stack = np.zeros(config.N_ultrasonics+1)
print(d_stack)
recording = True

# 操舵、駆動モーターの初期化
motor = motor.Motor()
motor.set_steer_pwm_duty(config.NUTRAL)
motor.set_throttle_pwm_duty(config.STOP)

# 超音波センサの初期化
# 別々にインスタンス化する例　ultrasonic_RrLH = ultrasonic.Ultrasonic("RrLH")
# 一気にインスタンス化
ultrasonics = {name: ultrasonic.Ultrasonic(name=name) for name in config.ultrasonics_list}
print(" 下記の", config.N_ultrasonics,"個の超音波センサを利用")
print(" ", config.ultrasonics_list)

# 操作判断プランナーの初期化
plan = planner.Planner("NoName")

# コントローラーの初期化
if config.CONTROLLER:
    joystick = joystick.Joystick()

# 一時停止（Enterを押すとプログラム実行開始）
print('Enterを押して走行開始!')
input()

# 途中でモータースイッチを切り替えたとき用に再度モーター初期化
# 初期化に成功するとピッピッピ！と３回音がなる、失敗時（PWMの値で約370-390以外の値が入りっぱなし）はピ...ピ...∞
motor.set_throttle_pwm_duty(config.STOP)

# 開始時間
start_time = time.perf_counter()

# ここから走行用プログラム
try:
    while True:
        # 認知（計測） ＃
        ## RrRHセンサ距離計測例：dis_RrRH = ultrasonic_RrRH.Mesure()
        ## 下記では一気に取得
        message = ""
        for i, name in enumerate(config.ultrasonics_list):
            d[i] = ultrasonics[name].Mesure()
            #message += name + ":" + str(round(ultrasonics[name].dis,2)).rjust(7, ' ') #Thony表示用にprint変更
            message += name + ":" + str(round(ultrasonics[name].dis,2))+ ", "
            # サンプリングレートを調整する場合は下記をコメントアウト外す
            #time.sleep(sampling_cycle)
        ## 記録と出力（タイムスタンプと距離データを配列に記録）
        if recording:
            d_stack = np.vstack((d_stack, np.insert(d, 0, time.perf_counter()-start_time)))
        print(message)

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
        ## モータードライバーに出力をセット
        motor.set_steer_pwm_duty(steer_pwm_duty)        
        motor.set_throttle_pwm_duty(throttle_pwm_duty)  
        ## 全体の状態を出力      
        print("Rec:",recording,"Str:",steer_pwm_duty,"Thr:",throttle_pwm_duty,end=' , ')

        ## ブレーキ
        if joystick.breaking:
            motor.breaking()

        # 停止処理 ＃
        plan.Stop(ultrasonics["Fr"])
        if plan.flag_stop ==True:
            ## 停止動作
            motor.set_steer_pwm_duty(config.NUTRAL)
            motor.set_throttle_pwm_duty(config.STOP)
            time.sleep(0.05)
            motor.set_throttle_pwm_duty(config.REVERSE)
            time.sleep(0.05)
            motor.set_throttle_pwm_duty(config.STOP)
            break

    # 終了処理
    GPIO.cleanup()
    header =""
    for name in config.ultrasonics_list:
        header += name + " "        
    np.savetxt(config.record_filename, d_stack, fmt='%.3e',header=header, comments="")
    print('記録停止')
    print("記録保存--> ",config.record_filename)
except KeyboardInterrupt:
    motor.set_steer_pwm_duty(config.NUTRAL)
    motor.set_throttle_pwm_duty(config.STOP)
    print('\nユーザーにより停止')
    GPIO.cleanup()
    header =""
    for name in config.ultrasonics_list:
        header += name + " "        
    np.savetxt(config.record_filename, d_stack, fmt='%.3e',header=header, comments="")
    print('記録停止')
    print("記録保存--> ",config.record_filename)
