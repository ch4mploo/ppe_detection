import cv2, time, asyncio, base64
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from detector import PPEDetector
from email_alert import send_alert

detector = PPEDetector()
camera = None
running = True
current_status = "No detection"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera, running
    print("Starting app")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Camera failed to open")
    running = True
    yield
    print("Stopping app")
    running = False
    camera.release()
    print("Camera released")

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/video")
async def websocket_video(ws: WebSocket):
    await ws.accept()
    print("Video WS connected")

    try:
        while running:
            success, frame = camera.read()
            if not success:
                await asyncio.sleep(0.1)
                continue

            status, results = detector.detect(frame)
            global current_status
            current_status = status

            annotated = frame.copy()
            annotated = results.plot(img=annotated)

            if detector.check_violation_timer(status):
                violation_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"violation_{violation_time}.jpg"
                cv2.imwrite(filename, annotated)
                send_alert(filename,violation_time)

            ret, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            await ws.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.03)  # ~30 FPS cap

    except WebSocketDisconnect:
        print("Video WS disconnected")

@app.websocket("/ws/status")
async def websocket_status(ws: WebSocket):
    await ws.accept()
    print("Status WS connected")

    try:
        while running:
            await ws.send_json({"status": current_status})
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        print("Status WS disconnected")

@app.get("/")
def dashboard():
    with open("templates/dashboard.html") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,                 # IMPORTANT on Raspberry Pi
        log_level="info",
        timeout_graceful_shutdown=5
    )
