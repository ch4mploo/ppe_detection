import cv2, time, threading
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from detector import PPEDetector
from email_alert import send_alert
from camera_manager import CameraManager

detector = PPEDetector()
camera = CameraManager(src=0)
current_status = "No detection"
shutdown_event = threading.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application...")
    camera.start()
    yield
    print("Shutting down application...")
    shutdown_event.set()
    camera.stop()


app = FastAPI(lifespan=lifespan)


def frame_generator(request: Request):
    global current_status

    while True:
        if shutdown_event.is_set():
            print("Stream exiting due to shutdown")
            return

        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        status, results = detector.detect(frame)
        current_status = status

        annotated = results.plot(img=frame.copy())

        # if detector.check_violat*ion_timer(status):
            # try:
            #     violation_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            #     filename = f"violation_{violation_time}.jpg"
            #     cv2.imwrite(filename, annotated)
            #     send_alert(filename,violation_time)
            # except Exception as e:
            #     print(f"Email failed: {e}")

        ret, buffer = cv2.imencode(".jpg", annotated)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

        time.sleep(0.03)  # ~30 FPS


@app.get("/")
def index():
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/video")
def video(request: Request):
    return StreamingResponse(
        frame_generator(request),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
def status():
    return {"status": current_status}
