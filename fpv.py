from flask import Flask, render_template, Response
from markupsafe import escape

#from camera import VideoCamera
import cv2
import camera_multiprocess

app = Flask(__name__)
data_sh = []

# "/" を呼び出したときには、indexが表示される。
@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    global data_sh
    while True:
        frame = camera.get_frame_multi()
        #print("print data_sh:",data_sh)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# returnではなくジェネレーターのyieldで逐次出力。
# Generatorとして働くためにgenとの関数名にしている
# Content-Type（送り返すファイルの種類として）multipart/x-mixed-replace を利用。
# HTTP応答によりサーバーが任意のタイミングで複数の文書を返し、紙芝居的にレンダリングを切り替えさせるもの。

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def run(*args ,**kwargs):
    global data_sh
    data_sh.append(args)
    print("print data_sh:",data_sh)
    app.run(**kwargs)    

class VideoCamera(object):
    def __init__(self):
        #self.video = cv2.VideoCapture(0)
        self.video = camera_multiprocess.VideoCaptureWrapper(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

        # read()は、二つの値を返すので、success, imageの2つ変数で受けています。
        # OpencVはデフォルトでは raw imagesなので JPEGに変換
        # ファイルに保存する場合はimwriteを使用、メモリ上に格納したい時はimencodeを使用
        # cv2.imencode() は numpy.ndarray() を返すので .tobytes() で bytes 型に変換

    def get_frame_multi(self):
        success, image = self.video.read()
        #image = cv2.resize(image, (160, 120))
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


if __name__ == '__main__':
    # 0.0.0.0はすべてのアクセスを受け付けます。    
    app.run(host='0.0.0.0', debug=True)
    

