import cv2
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from detector import PPEDetector
from email_alert import send_alert

app = FastAPI()
detector = PPEDetector()
camera = None
running = True
current_status = "No detection"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Docstring for lifespan
    
    :param app: Description
    :type app: FastAPI
    Context manager to handle application startup and shutdown.
     - On startup, initializes the camera.
     - On shutdown, releases the camera resource.
    """
    global camera, running

    print("Starting application...")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Failed to open camera")

    running = True

    yield # Application is running (yield will loop continually until shutdown)

    print("Shutting down application...")
    running = False

    if camera is not None:
        camera.release()
        print("Camera released")

app = FastAPI(lifespan=lifespan)

def generate_frames():
    global current_status, running, camera

    while running:
        try:
            success, frame = camera.read()

            if not success:
                print("Camera read failed, stopping stream")
                break

            status, results = detector.detect(frame)
            current_status = status

            annotated = frame.copy()
            annotated = results.plot(img=annotated)

            if detector.check_violation_timer(status):
                filename = f"violation_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated)
                send_alert(filename)

            ret, buffer = cv2.imencode(".jpg", annotated)
            if not ret:
                print("Frame encoding failed, skipping frame")
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        except Exception as e:
            print(f"Error in frame generation: {e}")
            break

    print("Frame generator stopped")


@app.get("/")
def index():
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())

@app.get("/video")
def video():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def status():
    return {"status": current_status}
