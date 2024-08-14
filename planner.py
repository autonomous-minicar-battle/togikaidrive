# coding:utf-8
import numpy as np
import config
import time
if config.mode_plan in ["NN","CNN"]:
    import torch.tensor
    from train_pytorch import denormalize_motor, normalize_ultrasonics 

class Planner:
    def __init__(self, name):
        self.name = name
        self.steer_pwm_duty =0
        self.throttle_pwm_duty = 0
        # 検知距離設定
        self.DETECTION_DISTANCE_Fr = config.DETECTION_DISTANCE_Fr
        self.DETECTION_DISTANCE_RL = config.DETECTION_DISTANCE_RL
        # 検知距離設定　他
        self.DETECTION_DISTANCE_STOP = config.DETECTION_DISTANCE_STOP
        self.DETECTION_DISTANCE_BACK = config.DETECTION_DISTANCE_BACK
        self.DETECTION_DISTANCE_TARGET = config.DETECTION_DISTANCE_TARGET
        self.DETECTION_DISTANCE_RANGE = config.DETECTION_DISTANCE_RANGE
        # PID用パラメータと
        self.K_P = config.K_P
        self.K_I = config.K_I
        self.K_D = config.K_D
        self.min_dis = 0

        #　判断フラグ
        self.flag_stop = False
        self.flag_back = False
        # 操作値出力
        self.message = ""
        # 過去の操作値記録回数
        self.records_steer_pwm_duty = np.zeros(config.motor_Nrecords)
        self.records_throttle_pwm_duty = np.zeros(config.motor_Nrecords)


    # 前側3センサーを用いた後退
    def Back(self, ultrasonic_Fr, ultrasonic_FrRH, ultrasonic_FrLH):
    ## 目前に前壁をtimes回検知
        times = 3
        # elifではなく、別のif文として評価
        if min(max(ultrasonic_Fr.records[:times]),max(ultrasonic_FrRH.records[:times]),max(ultrasonic_FrLH.records[:times])) < self.DETECTION_DISTANCE_BACK :
            self.flag_back = True
            print("後退")
        elif max(ultrasonic_Fr.records[:times]) > self.DETECTION_DISTANCE_BACK:
            self.flag_back = False  

    # 前側１センサーを用いた停止
    def Stop(self, ultrasonic_Fr):
        ## 目前に前壁をtimes回検知
        times = 3
        if max(ultrasonic_Fr.records[0:times-1]) < self.DETECTION_DISTANCE_STOP:
                self.flag_stop = True                
                print("停止")

    # 前側３センサーを用いた右左走行
    def Right_Left_3(self, dis_FrLH, dis_Fr, dis_FrRH):
        # 検知時の判断
        ## 壁を検知
        if dis_Fr < self.DETECTION_DISTANCE_Fr or dis_FrLH < self.DETECTION_DISTANCE_RL or dis_FrRH < self.DETECTION_DISTANCE_RL:
            ### 左＜右の距離
            if dis_FrLH < dis_FrRH :
                self.steer_pwm_duty =config.RIGHT
                self.throttle_pwm_duty = config.FORWARD_C
                self.message = "右旋回"
            ### 左＞右の距離
            else:
                self.steer_pwm_duty =config.LEFT
                self.throttle_pwm_duty = config.FORWARD_C
                self.message = "左旋回"            
        ## 前壁を検知なし
        else: 
            self.steer_pwm_duty =config.NUTRAL
            self.throttle_pwm_duty = config.FORWARD_S
            self.message = "直進中"

        ## モーターへ出力を返す
        if config.print_plan_result:
            print(self.message)
        return self.steer_pwm_duty, self.throttle_pwm_duty

    # 前側３センサーを用いた右左走行　過去の値でスムージング
    def Right_Left_3_Records(self, dis_FrLH, dis_Fr, dis_FrRH):
        self.steer_pwm_duty, self.throttle_pwm_duty  = self.Right_Left_3(dis_FrLH, dis_Fr, dis_FrRH)

        # 過去の値を記録の一番前に挿入し、最後を消す
        self.records_steer_pwm_duty = np.insert(self.records_steer_pwm_duty, 0, self.steer_pwm_duty)
        self.records_steer_pwm_duty = np.delete(self.records_steer_pwm_duty,-1)
        self.records_throttle_pwm_duty = np.insert(self.records_throttle_pwm_duty, 0, self.throttle_pwm_duty)
        self.records_throttle_pwm_duty = np.delete(self.records_throttle_pwm_duty,-1)

        ## モーターへ出力を返す
        if config.print_plan_result:
            print(self.message)
        return np.mean(self.records_steer_pwm_duty), np.mean(self.records_throttle_pwm_duty)

    # 右手法を用いた走行
    def RightHand(self, dis_FrRH, dis_RrRH):
        # 検知時の判断
        ## 右壁が遠い
        if dis_FrRH > self.DETECTION_DISTANCE_TARGET + self.DETECTION_DISTANCE_RANGE and dis_RrRH > self.DETECTION_DISTANCE_TARGET + self.DETECTION_DISTANCE_RANGE:
                self.steer_pwm_duty =config.RIGHT
                self.throttle_pwm_duty = config.FORWARD_C
                self.message = "右旋回"
        ## 右壁が近い
        elif dis_FrRH < self.DETECTION_DISTANCE_TARGET - self.DETECTION_DISTANCE_RANGE or dis_RrRH < self.DETECTION_DISTANCE_TARGET - self.DETECTION_DISTANCE_RANGE:
            self.steer_pwm_duty =config.LEFT
            self.throttle_pwm_duty = config.FORWARD_C
            self.message = "左旋回"            
        ## ちょうどよい
        else: 
            self.steer_pwm_duty =config.NUTRAL
            self.throttle_pwm_duty = config.FORWARD_S
            self.message = "直進中"

        ## モーターへ出力を返す
        if config.print_plan_result:
            print(self.message)
        return self.steer_pwm_duty, self.throttle_pwm_duty

    # 左手法を用いた走行
    def LeftHand(self, dis_FrLH, dis_RrLH):
        # 検知時の判断
        ## 左壁が遠い
        if dis_FrLH > self.DETECTION_DISTANCE_TARGET + self.DETECTION_DISTANCE_RANGE and dis_RrLH > self.DETECTION_DISTANCE_TARGET + self.DETECTION_DISTANCE_RANGE:
                self.steer_pwm_duty =config.LEFT
                self.throttle_pwm_duty = config.FORWARD_C
                self.message = "左旋回"
        ## 左壁が近い
        elif dis_FrLH < self.DETECTION_DISTANCE_TARGET - self.DETECTION_DISTANCE_RANGE or dis_RrLH < self.DETECTION_DISTANCE_TARGET - self.DETECTION_DISTANCE_RANGE:
            self.steer_pwm_duty =config.RIGHT
            self.throttle_pwm_duty = config.FORWARD_C
            self.message = "右旋回"            
        ## ちょうどよい
        else: 
            self.steer_pwm_duty =config.NUTRAL
            self.throttle_pwm_duty = config.FORWARD_S
            self.message = "直進中"

        ## モーターへ出力を返す
        if config.print_plan_result:
            print(self.message)
        return self.steer_pwm_duty, self.throttle_pwm_duty


    # 右手法のPIDを用いた走行
    def RightHand_PID(self, ultrasonic_FrRH, ultrasonic_RrRH,
        t=0,integral_delta_dis=0,min_dis=config.DETECTION_DISTANCE_TARGET):
        # 時間更新
        t_before = t
        t = time.perf_counter()
        delta_t = t-t_before
        # 右手法最小距離更新
        min_dis_before = min_dis
        min_dis = min(ultrasonic_FrRH.dis,ultrasonic_RrRH.dis)
        # 目標値までの差更新
        delta_dis = min_dis - self.DETECTION_DISTANCE_TARGET
        # 目標値までの差積分更新
        integral_delta_dis += delta_dis
         #速度更新
        v = (min_dis - min_dis_before)/delta_t
        # PID制御でステア値更新
        steer_pwm_duty_pid = self.K_P*delta_dis - self.K_D*v + self.K_I*integral_delta_dis 
        ### -100~100に収めて正の割合化
        steer_pwm_duty_pid = abs(max(-100,min(100,steer_pwm_duty_pid))/100)

        ## モーターへ出力を返す
        if config.print_plan_result:
            #print(self.message)
            print("output * PID:{:3.1f}, [P:{:3.1f}, I:{:3.1f}, D:{:3.1f}]".format(steer_pwm_duty_pid, self.K_P*delta_dis,self.K_D*v, self.K_I*integral_delta_dis))
        self.steer_pwm_duty, self.throttle_pwm_duty  = self.RightHand(ultrasonic_FrRH.dis, ultrasonic_RrRH.dis)
        return steer_pwm_duty_pid*self.steer_pwm_duty, self.throttle_pwm_duty

    # 左手法のPIDを用いた走行
    def LeftHand_PID(self, ultrasonic_FrLH, ultrasonic_RrLH,
        t=0,integral_delta_dis=0,min_dis=config.DETECTION_DISTANCE_TARGET):
        # 時間更新
        t_before = t
        t = time.perf_counter()
        delta_t = t-t_before
        # 右手法最小距離更新
        min_dis_before = min_dis
        min_dis = min(ultrasonic_FrLH.dis,ultrasonic_RrLH.dis)
        # 目標値までの差更新
        delta_dis = min_dis - self.DETECTION_DISTANCE_TARGET
        # 目標値までの差積分更新
        integral_delta_dis += delta_dis
         #速度更新
        v = (min_dis - min_dis_before)/delta_t
        # PID制御でステア値更新
        steer_pwm_duty_pid = self.K_P*delta_dis - self.K_D*v + self.K_I*integral_delta_dis 
        ### -100~100に収めて正の割合化
        steer_pwm_duty_pid = abs(max(-100,min(100,steer_pwm_duty_pid))/100)

        ## モーターへ出力を返す
        if config.print_plan_result:
            #print(self.message)
            print("output * PID:{:3.1f}, [P:{:3.1f}, I:{:3.1f}, D:{:3.1f}]".format(steer_pwm_duty_pid, self.K_P*delta_dis,self.K_D*v, self.K_I*integral_delta_dis))
        self.steer_pwm_duty, self.throttle_pwm_duty  = self.LeftHand(ultrasonic_FrLH.dis, ultrasonic_RrLH.dis)
        return steer_pwm_duty_pid*self.steer_pwm_duty, self.throttle_pwm_duty

    # Neural Netを用いた走行
    # train_pytorch.py内の正規化処理を用いる
    def NN(self, model, *args):
        ultrasonic_values = args
        input = normalize_ultrasonics(torch.tensor(ultrasonic_values, dtype=torch.float32).unsqueeze(0))
        output = denormalize_motor(model.predict(model, input).squeeze(0))
        self.steer_pwm_duty = int(output[0])
        self.throttle_pwm_duty = int(output[1])

        ## モーターへ出力を返す
        return self.steer_pwm_duty, self.throttle_pwm_duty
    
    def CNN(self, model, img):
        # train_pytorch.py内の正規化処理を用いる
        input = img
        output = denormalize_motor(model.predict(model, input).squeeze(0))
        self.steer_pwm_duty = int(output[0])
        self.throttle_pwm_duty = int(output[1])

        ## モーターへ出力を返す
        return self.steer_pwm_duty, self.throttle_pwm_duty

