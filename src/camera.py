import threading
from typing import Optional

import cv2
import numpy as np


class Camera:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self.error: Optional[str] = None
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self.error = None
        try:
            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        except cv2.error as error:
            raise RuntimeError(f"Could not initialize the webcam: {error}") from error
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                "Could not open the webcam. Check that it is connected and not in use by another app."
            )
        self._capture = capture
        self._stop_event.clear()
        self.running = True
        self._thread = threading.Thread(target=self._read_frames, daemon=True)
        self._thread.start()

    def read(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _read_frames(self) -> None:
        assert self._capture is not None
        while not self._stop_event.is_set():
            try:
                success, frame = self._capture.read()
            except cv2.error as error:
                self.error = f"Webcam read failed: {error}"
                break
            if not success:
                self.error = "The webcam stopped providing frames."
                break
            with self._frame_lock:
                self._latest_frame = frame
        self.running = False

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._thread = None
        self.running = False
        with self._frame_lock:
            self._latest_frame = None

