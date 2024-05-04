from flask import Flask, render_template, Response
import cv2
import multiprocessing
import camera_multiprocess

app = Flask(__name__)

frame_queue = multiprocessing.Queue()

def capture_frames(queue):
    #video = cv2.VideoCapture(0)
    video = camera_multiprocess.VideoCaptureWrapper(0)
    while True:
        success, frame = video.read()
        if success:
            queue.put(frame)
    video.release()

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        frame = frame_queue.get()
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    process = multiprocessing.Process(target=capture_frames, args=(frame_queue,))
    process.start()
    app.run(host='0.0.0.0', debug=True)
