import cv2
import time
import threading

class CameraManager:
    def __init__(self, src=0):
        self.src = src
        self.camera = None
        self.thread = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return

        self.camera = cv2.VideoCapture(self.src)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")

        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            success, frame = self.camera.read()
            if not success:
                time.sleep(0.1)
                continue

            with self.lock:
                self.frame = frame

        if self.camera:
            self.camera.release()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
