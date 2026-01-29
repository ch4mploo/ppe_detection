import time
import cv2
from ultralytics import YOLO

MODEL_PATH = "assets/models/best.pt"

# Object detection class
class PPEDetector:
    """
    Docstring for PPEDetector:
    Uses a custom-trained yolo model to detect PPE compliance in video frames.
    What is consider violating PPE:
    - No helmet (no_helmet)
    - No boots (no_boots)
    - No PPE at all (none)
    Methods:
    - detect(frame): Detects PPE compliance in the given frame.
    - check_violation_timer(status): Checks if a PPE violation has persisted for more than 3 seconds.
    Instance attributes:
    - model: The YOLO model loaded from the specified MODEL_PATH.
    - last_violation_time: Timestamp of the last detected PPE violation.
    - violation_triggered: Boolean flag indicating if a violation alert has been triggered.
    """
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.last_violation_time = None
        self.violation_triggered = False

    def detect(self, frame):
        results = self.model(frame, verbose=False, conf=0.35, iou=0.5, max_det=20)[0]

        labels = [self.model.names[int(cls)] for cls in results.boxes.cls]

        no_helmet = "no_helmet" in labels
        no_boots = "no_boots" in labels
        no_ppe = "none" in labels
        have_ppe = "helmet" in labels or "vest" in labels or "boots" in labels
        person = "Person" in labels or "person" in labels

        status = "No detection"

        if person:
            if (no_helmet or no_boots or no_ppe):
                status = "PPE not complied"
            elif have_ppe:
                status = "PPE complied"
            else:
                status = "Inconclusive"

        return status, results

    def check_violation_timer(self, status):
        now = time.time()

        if status == "PPE not complied":
            if self.last_violation_time is None:
                self.last_violation_time = now

            if now - self.last_violation_time >= 3:
                if not self.violation_triggered:
                    self.violation_triggered = True
                    return True
        else:
            self.last_violation_time = None
            self.violation_triggered = False

        return False
