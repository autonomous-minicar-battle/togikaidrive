# coding:utf-8
import time
import RPi.GPIO as GPIO
import config
import numpy as np

class Ultrasonic:
    def __init__(self, name):
        # 超音波センサ(HC-SR04)のクラス作成
        # データシート参考：https://cdn.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf
        # 超音波発信/受信用のGPiOピン番号
        self.name = name
        self.trig = config.ultrasonics_dict_trig[name]
        self.echo = config.ultrasonics_dict_echo[name]
        self.records = np.zeros(config.ultrasonics_Nrecords)
        self.distance = 0
        self.ultrasonicspeed = 343 #m/s
        self.cutoff = config.cutoff_distance
        self.cutofftime = self.cutoff/1000 /(self.ultrasonicspeed) #s
        
    # 障害物センサ測定関数
    def measure(self):
        self.distance, sigoff, sigon = 0, 0, 0
        # 10usのトリガー信号を送信
        GPIO.output(self.trig,GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig,GPIO.LOW)
        # エコー信号の立ち下がりと立ち上がりの時間を記録
        starttime=time.perf_counter()
        while(GPIO.input(self.echo)==GPIO.LOW):
            sigoff=time.perf_counter()
            if sigoff - starttime > 0.02: 
                break
        # エコー信号の立ち上がり時間が音速の往復時間
        while(GPIO.input(self.echo)==GPIO.HIGH):
            sigon=time.perf_counter()
            if sigon - sigoff > self.cutofftime: 
                break
        # time * sound speed / 2(round trip)
        d = int((sigon - sigoff) * self.ultrasonicspeed / 2 *1000)
        # 負値のノイズの場合は一つ前のデータに置き換え
        if d < 0:
            print("@",self.name,", a noise occureed, use the last value")
            self.distance = self.records[0]
            print(self.records)
        else:
            self.distance = d
        # 過去の超音波センサの値を記録の一番前に挿入し、最後を消す
        self.records = np.insert(self.records, 0, self.distance)
        self.records = np.delete(self.records,-1)
        return self.distance


class Ultrasonic_donkeycar:
    '''
    donkeycarで使う超音波センサのクラス作成。下記参考
    https://docs.donkeycar.com/parts/about/
    '''
    import logging
    logger = logging.getLogger("donkeycar.parts.ultrasonic")
    def __init__(self, ultrasonics_list, t_list, e_list, cutoff, poll_delay=0.0,batch_ms=50 ):
        import time
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(t_list,GPIO.OUT,initial=GPIO.LOW)
        GPIO.setup(e_list,GPIO.IN)

        self.cutoff = cutoff*2  # mm, round trip
        self.ultrasonicspeed = 343 #speed of sound in m/s
        self.cutofftime = self.cutoff/1000 /(self.ultrasonicspeed) #s 
        self.ultrasonics = []
        self.ultrasonics_list = ultrasonics_list
        self.t_list = t_list
        self.e_list = e_list
        self.logger.info(" USE ULTRASONIC SENSORS: ", self.ultrasonics_list)

        self.distances = [0]*len(ultrasonics_list) #a list of distance measurements
        self.distances_records = [[0]*len(ultrasonics_list) for i in range(3)]

        self.poll_delay = poll_delay
        self.measurement_batch_ms = batch_ms
        self.running = True
        self.on = True

        self.Nrecords = 3

    def poll(self):
        if self.running:
            try:
                for i in self.ultrasonics_list:
                    self.distances[i] = self.measure(i,self.t_list[i], self.e_list[i])
                self.distances_records[0:self.Nrecords-1]= self.distances_records[1:self.Nrecords]
                self.distances_records[self.Nrecords-1] = self.distances
                time.sleep(self.poll_delay)
            except Exception as e:
                self.logger.error("Error in ultrasonic poll() - {}".format(e))


    def measure(self, i, trig, echo):
        distance, sigoff, sigon = 0, 0, 0
        GPIO.output(trig,GPIO.HIGH)
        time.sleep(0.00001) #10us
        GPIO.output(trig,GPIO.LOW)
        starttime=time.perf_counter()
        while(GPIO.input(echo)==GPIO.LOW):
            sigoff=time.perf_counter()
            if sigoff - starttime > 0.0005:
            #     print("break1")
                break
        sigoff=time.perf_counter()
        while(GPIO.input(echo)==GPIO.HIGH):
            sigon=time.perf_counter()
            # more than 0.06s suggested for next cycle of 1 sensor
            if sigon - sigoff > self.cutofftime: 
            #     print("break2")
                break
        # time * sound speed / 2(round trip)
        d = int((sigon - sigoff)/1000 * self.ultrasonicspeed / 2)
        # noise data replaced by old data
        if d < 0:
            self.logger.warning("@",self.ultrasonics_list[i],", a noise occureed, use the last value")
            distance = self.distances_records[-1][i]
        else:
            distance = d
        return distance

    def update(self):
        while self.running:
            self.poll()
            time.sleep(0)  # yield time to other threads

    def run_threaded(self):
        if self.running:
            return self.distances
        return []
    
    def run(self):
        if not self.running:
            return []
        batch_time = time.time() + self.measurement_batch_ms / 1000.0
        while True:
            self.poll()
            time.sleep(0)  # yield time to other threads
            if time.time() >= batch_time:
                break
        return self.distances
    
    def shutdown(self):
        self.running = False
        GPIO.cleanup()
        time.sleep(0.5)
        
        
if __name__ == "__main__":
    import config
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    ## GPIOピン番号の指示方法
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(config.e_list,GPIO.IN)
    GPIO.setup(config.t_list,GPIO.OUT,initial=GPIO.LOW)

    # 超音波センサを設定、使う分だけリストにultrasonicインスタンスを入れる
    #ultraconic = Ultraconic(config.t_list[0],config.e_list[0]) 
    ultrasonics = [] 
    # 一つだけ使う場合、複数使う場合はコメントアウト外す
    #config.ultrasonics_list = ["Fr"]
    config.ultrasonics_list = ["RrLH", "FrLH", "Fr", "FrRH","RrRH"]
    for name in config.ultrasonics_list:
        ultrasonics.append(Ultrasonic(name))

    print(" 下記の超音波センサを利用")
    print(" ", ultrasonics)

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
                message += str(config.ultrasonics_list[j]) + ":" + str(dis)+ ", "
                time.sleep(sampling_cycle)
            d_stack = np.vstack((d_stack, np.insert(d, 0, time.perf_counter()-start_time)))
            print(message)
        GPIO.cleanup()
        np.savetxt(config.record_filename, d_stack, fmt='%.3e')
        ## 列方向に時間平均: np.round axis=0、スライスで時間の列は取得しない[:,1:]
        print('測定回数： ',sampling_times)
        print('平均距離：', np.round(np.mean(d_stack[:,1:], axis=0),0))
        print("平均測定時間/センサ(秒):",round((time.perf_counter()-start_time)/sampling_times/len(ultrasonics),2))
        print("記録保存--> ",config.record_filename)

    except KeyboardInterrupt:
        np.savetxt(config.record_filename, d_stack, fmt='%.3e')
        print('stop!')
        GPIO.cleanup()
