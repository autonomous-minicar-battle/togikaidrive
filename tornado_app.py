import cv2
import time
import asyncio
import tornado
import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web

from capture_video import CaptureVideo

global port
port  = 5000
def capture_frame():
    CaptureVideo.capture_start()

class HttpHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def initialize(self):
        pass

    def get(self):
        self.render('tornado_index.html')

class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print(self.request.remote_ip, ": connection opened")
        self.ioloop = tornado.ioloop.IOLoop.current()
        self.loop()

    def on_close(self):
        print("Session closed")
        self.close()

    def check_origin(self, origin):
        return True

    def loop(self):
        self.ioloop.add_timeout(time.time() + 0.1, self.loop)
        frame = CaptureVideo.get_frame()
        #
        frame = cv2.resize(frame, (160, 120))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),100]
        ret, decimg = cv2.imencode('.jpg', frame, encode_param)
        if self.ws_connection:
            if ret:
                message = decimg.tobytes()
                self.write_message(message, binary=True)

def main():
    try:
        app = tornado.web.Application([
            (r'/', HttpHandler),
            (r'/camera', WSHandler),
        ])
        http_server = tornado.httpserver.HTTPServer(app)
        http_server.listen(port)

        print('server start')
        io_loop = tornado.ioloop.IOLoop.current()
        capture_frame()
        io_loop.start()

    except KeyboardInterrupt:
        print('server stop')
        ioloop = tornado.ioloop.IOLoop.current()
        ioloop.stop()
        ioloop.close()

if __name__=='__main__':
    main()