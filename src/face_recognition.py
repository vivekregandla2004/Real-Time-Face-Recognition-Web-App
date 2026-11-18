from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .database import Identity
from .face_detection import DETECTOR_MODEL, YuNetDetector
from .utils import MODEL_DIR


RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"
COSINE_THRESHOLD = 0.363


@dataclass
class FaceObservation:
    box: tuple[int, int, int, int]
    aligned_face: np.ndarray
    embedding: np.ndarray


@dataclass
class Recognition:
    box: tuple[int, int, int, int]
    name: str
    similarity: Optional[float]


def match_embedding(
    query: np.ndarray,
    identities: Sequence[Identity],
    threshold: float = COSINE_THRESHOLD,
) -> tuple[Optional[str], Optional[float]]:
    query_vector = np.asarray(query, dtype=np.float32).reshape(-1)
    query_norm = float(np.linalg.norm(query_vector))
    if query_vector.shape != (128,) or query_norm == 0:
        return None, None
    query_vector = query_vector / query_norm

    best_name: Optional[str] = None
    best_similarity: Optional[float] = None
    for identity in identities:
        for stored_embedding in identity.embeddings:
            stored = np.asarray(stored_embedding, dtype=np.float32).reshape(-1)
            if stored.shape != (128,):
                continue
            stored_norm = float(np.linalg.norm(stored))
            if stored_norm == 0:
                continue
            similarity = float(np.dot(query_vector, stored / stored_norm))
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_name = identity.name

    if best_similarity is None or best_similarity < threshold:
        return None, best_similarity
    return best_name, best_similarity


class FaceAnalyzer:
    def __init__(self, model_dir: Path = MODEL_DIR):
        detector_path = Path(model_dir) / DETECTOR_MODEL
        recognizer_path = Path(model_dir) / RECOGNIZER_MODEL
        missing = [path.name for path in (detector_path, recognizer_path) if not path.is_file()]
        if missing:
            names = ", ".join(missing)
            raise FileNotFoundError(
                f"Required OpenCV Zoo model file(s) are missing from {model_dir}: {names}. "
                "See the README for model download instructions."
            )

        self.detector = YuNetDetector(detector_path)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def extract(self, frame: np.ndarray) -> list[FaceObservation]:
        height, width = frame.shape[:2]
        faces = self.detector.detect(frame)

        observations: list[FaceObservation] = []
        for face in faces:
            aligned = self.recognizer.alignCrop(frame, face)
            embedding = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
            x, y, box_width, box_height = (int(round(value)) for value in face[:4])
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            box_width = max(1, min(box_width, width - x))
            box_height = max(1, min(box_height, height - y))
            observations.append(
                FaceObservation(
                    box=(x, y, box_width, box_height),
                    aligned_face=aligned,
                    embedding=embedding,
                )
            )
        return observations

    def recognize(
        self,
        frame: np.ndarray,
        identities: Sequence[Identity],
    ) -> list[Recognition]:
        results: list[Recognition] = []
        for observation in self.extract(frame):
            name, similarity = match_embedding(observation.embedding, identities)
            results.append(
                Recognition(
                    box=observation.box,
                    name=name or "UNKNOWN",
                    similarity=similarity,
                )
            )
        return results
