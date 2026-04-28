import cv2
from ultralytics import YOLO
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    center_x: int
    center_y: int

    def to_bbox(self) -> tuple[float, float, float, float]:
        """Trả về (x1, y1, x2, y2) để tính toán 3D"""
        return (self.x1, self.y1, self.x2, self.y2)

    def width(self) -> float:
        return self.x2 - self.x1

    def height(self) -> float:
        return self.y2 - self.y1


class HelipadDetector:
    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.6,
        imgsz: int = 640,
    ):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self._prev_time = 0.0
        self.fps = 0.0

    def detect(self, frame) -> Optional[Detection]:
        """
        Nhận vào 1 frame (numpy array BGR).
        Trả về Detection đầu tiên có confidence cao nhất, hoặc None nếu không tìm thấy.
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            stream=True,
            verbose=False,
        )

        best: Optional[Detection] = None

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()

                if best is None or conf > best.conf:
                    best = Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        conf=conf,
                        center_x=int((x1 + x2) / 2),
                        center_y=int((y1 + y2) / 2),
                    )

        return best

    def detect_all(self, frame) -> list[Detection]:
        """
        Trả về danh sách tất cả Detection trong frame.
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            stream=True,
            verbose=False,
        )

        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    conf=conf,
                    center_x=int((x1 + x2) / 2),
                    center_y=int((y1 + y2) / 2),
                ))

        return detections

    def draw(self, frame, detection: Detection) -> None:
        """Vẽ bbox + center + label lên frame (in-place)."""
        x1, y1, x2, y2 = int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (detection.center_x, detection.center_y), 1, (0, 0, 255), -1)
        # label = f"Helipad {detection.conf:.2f}"
        # cv2.putText(frame, label, (x1, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def update_fps(self) -> float:
        curr = time.time()
        self.fps = 1.0 / (curr - self._prev_time) if self._prev_time else 0.0
        self._prev_time = curr
        return self.fps