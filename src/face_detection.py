from pathlib import Path

import cv2
import numpy as np


DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"


class YuNetDetector:
    def __init__(self, model_path: Path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"YuNet model file was not found: {model_path}")
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold=0.85,
            nms_threshold=0.3,
            top_k=5000,
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)
