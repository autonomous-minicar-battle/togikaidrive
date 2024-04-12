# coding:utf-8
import Adafruit_PCA9685
import config
import time

class Motor:
    def __init__(self):
        self.pwm = Adafruit_PCA9685.PCA9685(address=0x40)
        self.pwm.set_pwm_freq(60)
        self.CHANNEL_STEERING = config.CHANNEL_STEERING
        self.CHANNEL_THROTTLE = config.CHANNEL_THROTTLE
        self.STEERING_CENTER_PWM = config.STEERING_CENTER_PWM
        self.STEERING_WIDTH_PWM = config.STEERING_WIDTH_PWM
        self.STEERING_RIGHT_PWM = config.STEERING_RIGHT_PWM
        self.STEERING_LEFT_PWM = config.STEERING_LEFT_PWM
        self.THROTTLE_STOPPED_PWM = config.THROTTLE_STOPPED_PWM
        self.THROTTLE_WIDTH_PWM = config.THROTTLE_WIDTH_PWM
        self.THROTTLE_FORWARD_PWM = config.THROTTLE_FORWARD_PWM
        self.THROTTLE_REVERSE_PWM = config.THROTTLE_REVERSE_PWM

    def set_throttle_pwm_duty(self, duty):
        if duty >= 0:
            throttle_pwm = int(self.THROTTLE_STOPPED_PWM + (self.THROTTLE_FORWARD_PWM - self.THROTTLE_STOPPED_PWM) * duty / 100)
        else:
            throttle_pwm = int(self.THROTTLE_STOPPED_PWM + (self.THROTTLE_REVERSE_PWM - self.THROTTLE_STOPPED_PWM) * abs(duty) / 100)
        
        self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, throttle_pwm)
        #print(throttle_pwm)

    def set_steer_pwm_duty(self, duty):
        if duty >= 0:
            steer_pwm = int(self.STEERING_CENTER_PWM + (self.STEERING_RIGHT_PWM - self.STEERING_CENTER_PWM) * duty / 100)
        else:
            steer_pwm = int(self.STEERING_CENTER_PWM + (self.STEERING_LEFT_PWM - self.STEERING_CENTER_PWM) * abs(duty) / 100)
        steer_pwm = self.limit_steer_PWM(steer_pwm)
        #    print ("Caution!, please set 260~450 not to break!\n")
        #else:
        self.pwm.set_pwm(self.CHANNEL_STEERING, 0, steer_pwm)
        #print(steer_pwm)
        
    def limit_steer_PWM(self,steer_pwm):
        if steer_pwm > config.STEERING_RIGHT_PWM_LIMIT:
            print ("\n!!!警告!!! 壊さないように最大値:{}で設定ください!\n".format(config.STEERING_RIGHT_PWM_LIMIT))
            return config.STEERING_RIGHT_PWM_LIMIT
        elif steer_pwm < config.STEERING_LEFT_PWM_LIMIT:
            print ("\n!!!警告!!! 壊さないように最小値:{}で設定ください!\n".format(config.STEERING_LEFT_PWM_LIMIT))
            return config.STEERING_LEFT_PWM_LIMIT
        else:
            return steer_pwm
        

    def adjust_steering(self):
        print('========================================')
        print(' ステアリング調整、ステアの中心位置を決める')
        print('========================================')
        while True:
            print('PWM の値を入力, 例 390')
            print('中心値が決まればEnter')
            print('ジジっとノイズがし続けたら注意、壊れる、、、')
            ad = input()
            if ad == 'e' or ad =='':
                self.STEERING_RIGHT_PWM = self.STEERING_CENTER_PWM + self.STEERING_WIDTH_PWM
                self.STEERING_LEFT_PWM = self.STEERING_CENTER_PWM - self.STEERING_WIDTH_PWM
                break
            self.STEERING_CENTER_PWM = int(ad)
            self.limit_steer_PWM(self.STEERING_CENTER_PWM)
            self.pwm.set_pwm(self.CHANNEL_STEERING, 0, self.STEERING_CENTER_PWM)
        print('')
        return self.STEERING_RIGHT_PWM,self.STEERING_CENTER_PWM,self.STEERING_LEFT_PWM

    def adjust_throttle(self):
        print('========================================')
        print(' スロットル調整、ニュートラル位置を決める')
        print('========================================')
        while True:
            print('PWM の値を入力, 例 390')
            print('中心値が決まればEnter')
            ad = input()
            if ad == 'e' or ad =='':
                self.THROTTLE_FORWARD_PWM = self.THROTTLE_STOPPED_PWM + self.THROTTLE_WIDTH_PWM
                self.THROTTLE_REVERSE_PWM = self.THROTTLE_STOPPED_PWM - self.THROTTLE_WIDTH_PWM
                break
            self.THROTTLE_STOPPED_PWM = int(ad)
            self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_STOPPED_PWM)
        print('')
        return self.THROTTLE_FORWARD_PWM,self.THROTTLE_STOPPED_PWM,self.THROTTLE_REVERSE_PWM

    def writetofile(self, path):
        with open(path, 'w') as f:
            f.write(f'STEERING_RIGHT_PWM = {self.STEERING_RIGHT_PWM}\n')
            f.write(f'STEERING_CENTER_PWM = {self.STEERING_CENTER_PWM}\n')
            f.write(f'STEERING_LEFT_PWM = {self.STEERING_LEFT_PWM}\n')
            f.write(f'THROTTLE_FORWARD_PWM = {self.THROTTLE_FORWARD_PWM}\n')
            f.write(f'THROTTLE_STOPPED_PWM = {self.THROTTLE_STOPPED_PWM}\n')
            f.write(f'THROTTLE_REVERSE_PWM = {self.THROTTLE_REVERSE_PWM}\n')

    def breaking(self):
            print(" breaking!!!")
            self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_STOPPED_PWM)
            time.sleep(0.05)
            self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_REVERSE_PWM)
            time.sleep(0.05)
            self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_STOPPED_PWM)
            time.sleep(0.05)
            self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_REVERSE_PWM)
            time.sleep(0.05)
            self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_STOPPED_PWM)


if __name__ == "__main__":
    try:
        motor = Motor()
        motor.set_throttle_pwm_duty(0)
        motor.set_steer_pwm_duty(0)

        STEERING_RIGHT_PWM,STEERING_CENTER_PWM,STEERING_LEFT_PWM = motor.adjust_steering()
        THROTTLE_FORWARD_PWM,THROTTLE_STOPPED_PWM,THROTTLE_REVERSE_PWM = motor.adjust_throttle()
        print("---下記をconfig.pyの値に入力。\n値の微調整は走りながら決定。")
        print(f'STEERING_RIGHT_PWM = {STEERING_RIGHT_PWM}')
        print(f'STEERING_CENTER_PWM = {STEERING_CENTER_PWM}')
        print(f'STEERING_LEFT_PWM = {STEERING_LEFT_PWM}')
        print(f'THROTTLE_FORWARD_PWM = {THROTTLE_FORWARD_PWM}')
        print(f'THROTTLE_STOPPED_PWM = {THROTTLE_STOPPED_PWM}')
        print(f'THROTTLE_REVERSE_PWM = {THROTTLE_REVERSE_PWM}')
        print("---上記をconfig.pyの値に入力。\n値の微調整は走りながら決定。")
        motor.set_throttle_pwm_duty(0)
        motor.set_steer_pwm_duty(0)

    #path = "config.py"
    #motor.writetofile(path)
    #print(path, motor.STEERING_RIGHT_PWM, motor.STEERING_CENTER_PWM, motor.STEERING_LEFT_PWM, motor.THROTTLE_FORWARD_PWM, motor.THROTTLE_STOPPED_PWM, motor.THROTTLE_REVERSE_PWM)

    except KeyboardInterrupt:
        print('ユーザーにより停止')
        motor.set_steer_pwm_duty(config.NUTRAL)
        motor.set_throttle_pwm_duty(config.STOP)
