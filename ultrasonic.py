# coding:utf-8
import time
import RPi.GPIO as GPIO
import config
import numpy as np

class Ultrasonic:
    def __init__(self, name):
        # 超音波発信/受信用のGPiOピン番号
        self.name = name
        self.trig = config.ultrasonics_dict_trig[name]
        self.echo = config.ultrasonics_dict_echo[name]
        self.records = np.zeros(config.ultrasonics_Nrecords)
        self.dis = 0

    # 障害物センサ測定関数
    def measure(self):
        self.dis = 0
        sigoff = 0
        sigon = 0
        GPIO.output(self.trig,GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig,GPIO.LOW)
        starttime=time.perf_counter()
        while(GPIO.input(self.echo)==GPIO.LOW):
            sigoff=time.perf_counter()
            if sigoff - starttime > 0.02: 
            #     print("break1")
                break
        while(GPIO.input(self.echo)==GPIO.HIGH):
            sigon=time.perf_counter()
            if sigon - sigoff > 0.02: 
            #     print("break2")
                break
        # time * sound speed / 2(round trip)
        d = (sigon - sigoff)*340000/2
        # 2m以上は無視
        if d > 2000:
            self.dis = 2000
            #print("more than 2m!")
        # 負値のノイズの場合は一つ前のデータに置き換え
        elif d < 0:
            print("@",self.name,", a noise occureed, use the last value")
            self.dis = self.records[0]
            print(self.records)
        else:
            self.dis = d
        # 過去の超音波センサの値を記録の一番前に挿入し、最後を消す
        self.records = np.insert(self.records, 0, self.dis)
        self.records = np.delete(self.records,-1)
        return self.dis

        
if __name__ == "__main__":
    # 超音波センサを設定、使う分だけリストにultrasonicインスタンスを入れる
    #ultraconic = Ultraconic(config.t_list[0],config.e_list[0]) 
    ultrasonics = [] 
    for name in config.ultrasonics_list:
        ultrasonics.append(Ultrasonic(name))
    print(" 下記の", config.N_ultrasonics,"個の超音波センサを利用")
    print(" ", config.ultrasonics_list)

    # データ記録用配列作成
    d = np.zeros(len(ultrasonics))
    d_stack = np.zeros(len(ultrasonics)+1)
    # 記録用開始時間
    start_time = time.perf_counter()
    # 記録回数
    sampling_times = config.sampling_times
    # 目標サンプルレート、複数センサ利用の場合は調整
    sampling_cycle = config.sampling_cycle/len(ultrasonics)

    # 一時停止（Enterを押すとプログラム実行開始）
    print('計測回数を入力し、Enterで計測開始')
    print('Enterのみのデフォルト：{}'.format(sampling_times))
    #　入力の確認
    while True:
        sampling_times = input()
        if sampling_times.isnumeric() and int(sampling_times) > 0:
            break
        elif sampling_times == "":
            sampling_times = config.sampling_times
            break
        else:
            print("1以上の整数を入力...")
    print('{}回、計測開始します!'.format(sampling_times))
    # 入力はintに戻しておく
    sampling_times = int(sampling_times)
    
    try:
        for i in range(sampling_times):
            message = ""
            for j in range(len(ultrasonics)):
                dis = ultrasonics[j].measure()
                #距離データを配列に記録
                d[j] = dis
                #表示用にprint変更
                #message += str(config.ultrasonics_list[j]) + ":" + str(round(dis,2)).rjust(7, ' ') 
                message += str(config.ultrasonics_list[j]) + ":" + str(round(dis,2))+ ", "
                time.sleep(sampling_cycle)
            d_stack = np.vstack((d_stack, np.insert(d, 0, time.perf_counter()-start_time)))
            print(message)
        GPIO.cleanup()
        np.savetxt(config.record_filename, d_stack, fmt='%.3e')
        ## 列方向に時間平均: np.round axis=0、スライスで時間の列は取得しない[:,1:]
        print('測定回数： ',sampling_times)
        print('平均距離：', np.round(np.mean(d_stack[:,1:], axis=0),4))
        print("平均測定時間/センサ(秒):",round((time.perf_counter()-start_time)/sampling_times/len(ultrasonics),2))
        print("記録保存--> ",config.record_filename)

    except KeyboardInterrupt:
        np.savetxt(config.record_filename, d_stack, fmt='%.3e')
        print('stop!')
        GPIO.cleanup()
