import asyncio
import cv2

class CaptureVideo:
    is_running = False
    frame = None

    @classmethod
    def capture_start(cls, camera_id=0):
        if not cls.is_running:
            cls.is_running = True
            asyncio.ensure_future(cls.__capture_loop(camera_id))

    @classmethod
    def get_frame(cls):
        if cls.is_running:
            return cls.frame
        else:
            print('capture loop is not running!')

    @classmethod
    def stop_capture(cls):
        if cls.is_running:
            cls.is_running = False

    @classmethod
    async def __capture_loop(cls, camera_id):
        cap = cv2.VideoCapture(camera_id)
        try:
            while cls.is_running:
                ret, cls.frame = cap.read()
                await asyncio.sleep(0.1)
        except:
            loop = asyncio.get_event_loop()
            loop.stop()
            cls.isrunning = False
            cap.release()