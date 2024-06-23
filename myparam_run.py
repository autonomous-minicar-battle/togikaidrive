# coding:utf-8
if not __name__ == "__main__":
    import config
    # ~~~出前授業用に一部のバラメータを変更
    # ！！！出前授業用に生徒が変更するバラメータ　ここから ！！！　#
    # モーター出力パラメータ （デューティー比：-100~100で設定）
    # スロットル用
    config.FORWARD_S = 80 #50 #ストレートでの値, joy_accel1
    config.FORWARD_C = 50 #40 #カーブでのの値, joy_accel2
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
    config.DETECTION_DISTANCE_RL = 500
    ### 他
    config.DETECTION_DISTANCE_TARGET = 130 #180 #目標距離
    config.DETECTION_DISTANCE_RANGE = 60/2 #修正認知半径距離

    # 判断モード選択
    ##　選択肢："Right_Left_3","Right_Left_3_Records","RightHand","RightHand_PID"
    config.mode_plan = "Right_Left_3" #"GoStraight"

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
    config.board = "old" #old：~2023年たこ足配線、new：新ボード

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
    config.STEERING_CENTER_PWM = 360
    config.STEERING_WIDTH_PWM = 80
    config.STEERING_RIGHT_PWM = config.STEERING_CENTER_PWM + config.STEERING_WIDTH_PWM
    config.STEERING_LEFT_PWM = config.STEERING_CENTER_PWM - config.STEERING_WIDTH_PWM
    ### !!!ステアリングを壊さないための上限下限の値設定  
    config.STEERING_RIGHT_PWM_LIMIT = 550
    config.STEERING_LEFT_PWM_LIMIT = 250

    ## アクセルのPWM値(motor.pyで調整した後値を入れる)
    ## モーターの回転音を聞き、音が変わらないところが最大/最小値とする
    config.THROTTLE_STOPPED_PWM = 370 #370-390当たりにニュートラル値がある
    config.THROTTLE_FORWARD_PWM = 500
    config.THROTTLE_REVERSE_PWM = 240

    ## thonny plotterで値を見るときにTrue
    config.plotter = True
    ## 使わないライブラリOFF
    config.HAVE_CAMERA =False
    config.HAVE_IMU =False
    config.HAVE_CONTROLLER = False
    config.fpv = False
    # ~~~出前授業用に一部のバラメータを変更　ここまで ~~~

# 以下の順番でモジュールを実行していく
# myparam_run.py➔run.py➔config➔myparamのconfig
if __name__ == "__main__":
    import run
