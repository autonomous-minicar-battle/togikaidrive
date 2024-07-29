from __future__ import annotations

import ctypes
import multiprocessing
import multiprocessing.sharedctypes
import multiprocessing.synchronize
import signal
from time import perf_counter, time

from typing import cast

import cv2
import numpy as np

import config

# ビデオのキャプチャとバッファの更新を担当する関数
def _update(args: tuple, buffer: ctypes.Array[ctypes.c_uint8], ready: multiprocessing.synchronize.Event, cancel: multiprocessing.synchronize.Event):

    # キャプチャ中にCtrl+Cが押された場合のシグナルハンドラの設定
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # キャプチャデバイスをオープン
    video_capture = cv2.VideoCapture(*args)
    if not video_capture.isOpened():
        raise IOError()

    try:
        # キャプチャループ
        while not cancel.is_set():
            # フレームの取得
            ret, img = cast("tuple[bool, cv2.Mat]", video_capture.read())
            if not ret:
                continue

            # バッファ更新の準備
            ready.clear()
            memoryview(buffer).cast('B')[:] = memoryview(img).cast('B')[:]
            ready.set()

    finally:
        # キャプチャデバイスの解放
        video_capture.release()


# ビデオの情報取得を担当する関数
def _get_information(args: tuple):

    # キャプチャデバイスをオープン
    video_capture = cv2.VideoCapture(*args)
    if not video_capture.isOpened():
        raise IOError()

    try:
        # 最初のフレームの取得
        ret, img = cast("tuple[bool, cv2.Mat]", video_capture.read())
        if not ret:
            raise IOError()

        # フレームの形状を返す
        return img.shape

    finally:
        # キャプチャデバイスの解放
        video_capture.release()


# ビデオキャプチャをラップするクラス
class VideoCaptureWrapper:

    def __init__(self, *args) -> None:
        self.currentframe = None

        # キャプチャデバイスの情報取得
        self.__shape = _get_information(args)
        height, width, channels = self.__shape

        # 共有バッファの作成
        self.__buffer = multiprocessing.sharedctypes.RawArray(
            ctypes.c_uint8, height * width * channels)

        # 同期用イベントの作成
        self.__ready = multiprocessing.Event()
        self.__cancel = multiprocessing.Event()

        # キャプチャプロセスの開始
        self.__enqueue = multiprocessing.Process(target=_update, args=(
            args, self.__buffer, self.__ready, self.__cancel), daemon=True)
        self.__enqueue.start()

        self.__released = cast(bool, False)

    # フレームの取得
    def read(self):
        self.__ready.wait()
        self.currentframe = np.reshape(self.__buffer, self.__shape).copy()
        return cast(bool, True), self.currentframe

    # キャプチャの解放
    def release(self):
        if self.__released:
            return

        self.__cancel.set()
        self.__enqueue.join()
        self.__released = True

    # デストラクタ
    def __del__(self):
        try:
            self.release()
        except:
            pass

    #def save(self, img, img_sh, ts, steer, throttle,  image_dir):
    def save(self, img,  ts, steer, throttle,  image_dir, img_size_w, img_size_h):
        try:
            img = cv2.resize(img, (img_size_w, img_size_h))
            cv2.imwrite(image_dir +'/' + ts +'_'+ str(steer) +'_'+ str(throttle) +'.jpg', img)            
            #img_sh[:] = img.flatten()
            return img
        except:
            print("Cannot save image!")
            pass
        
if __name__ == "__main__":
    print(" 注意: もしカメラが既に起動中で赤色LEDがオン、resource busyになる場合は再起動！\n")
    # カメラを使ってVideoCaptureWrapperを作成
    video_capture = VideoCaptureWrapper(0)
    _, img = video_capture.read()
    img = cv2.resize(img, (160, 120))
    # 一枚だけ保存
    #video_capture.save(img, time(), "steerpwm", "throttlepwm", ".")

    try:
        # チェック用
        #while True:
        #    print(video_capture.currentframe)

        # メインループ
        while True:
            start_time = perf_counter()
            _, img = video_capture.read()
            #img = cv2.resize(img, (160, 120))
            print( "fps:" + str(round(1/(perf_counter() - start_time))))
            try:
                #cv2.imshow("window", img)
                #cv2.waitKey(1)
                pass
            # ローカル画面出力無しの場合
            except :
                pass

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt, camera stopping")
        pass

    # キャプチャの解放
    video_capture.release()
