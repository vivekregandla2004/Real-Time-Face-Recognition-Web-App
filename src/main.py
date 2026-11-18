import logging
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np

import cv2
from PIL import Image, ImageTk

from .camera import Camera
from .database import DataStoreError, FaceDatabase, Identity
from .face_recognition import FaceAnalyzer, Recognition
from .face_registration import FaceRegistration
from .utils import ensure_project_directories


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class FaceRecognitionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Local Face Recognition")
        self.root.geometry("1020x760")
        self.root.minsize(800, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        ensure_project_directories()
        self.database = FaceDatabase()
        self.registration = FaceRegistration(self.database)
        self.camera = Camera()
        self.identities: list[Identity] = []
        self.analyzer: Optional[FaceAnalyzer] = None
        self.camera_active = False
        self.last_inference = 0.0
        self.recognitions: list[Recognition] = []
        self.preview_image: Optional[ImageTk.PhotoImage] = None

        self.status_text = tk.StringVar(value="Starting application...")
        self.people_text = tk.StringVar(value="Recognized: —")
        self.similarity_text = tk.StringVar(value="Similarity: —")
        self._build_interface()
        self._load_local_data()
        self.root.after(30, self._update_preview)

    def _build_interface(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            container,
            text="Real-Time Face Recognition",
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor=tk.W, pady=(0, 10))

        self.preview = ttk.Label(
            container,
            text="Start the camera to see the live preview",
            anchor=tk.CENTER,
            background="#20252b",
            foreground="white",
        )
        self.preview.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        ttk.Label(container, textvariable=self.status_text).pack(anchor=tk.W)
        ttk.Label(
            container,
            textvariable=self.people_text,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor=tk.W, pady=(7, 0))
        ttk.Label(container, textvariable=self.similarity_text).pack(anchor=tk.W, pady=(2, 10))

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X)
        self.start_button = ttk.Button(controls, text="Start Camera", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_button = ttk.Button(
            controls,
            text="Stop Camera",
            command=self.stop_camera,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 8))
        self.register_button = ttk.Button(
            controls,
            text="Register Face",
            command=self.open_registration,
        )
        self.register_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Exit", command=self.close).pack(side=tk.RIGHT)

        ttk.Label(
            container,
            text="Local processing only. Register people only with their permission.",
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(12, 0))

    def _load_local_data(self) -> None:
        try:
            self.analyzer = FaceAnalyzer()
        except (OSError, cv2.error) as error:
            self.status_text.set(f"Recognition models unavailable: {error}")
            self.start_button.configure(state=tk.DISABLED)
            logging.warning("Could not initialize recognition models: %s", error)
            return

        try:
            self.identities = self.database.load_identities()
            self.status_text.set(
                f"Ready. {len(self.identities)} registered identity/identities loaded."
            )
        except DataStoreError as error:
            self.status_text.set(f"Registration data could not be loaded: {error}")
            messagebox.showwarning(
                "Registration data",
                f"The saved registration data could not be loaded.\n\n{error}",
                parent=self.root,
            )

    def start_camera(self) -> None:
        if self.analyzer is None:
            messagebox.showerror(
                "Recognition unavailable",
                "The YuNet/SFace models are not ready. Follow the model setup instructions in README.md.",
                parent=self.root,
            )
            return
        try:
            self.camera.start()
        except RuntimeError as error:
            self.status_text.set(str(error))
            messagebox.showerror("Camera error", str(error), parent=self.root)
            return
        self.camera_active = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_text.set("Camera running. Looking for faces...")

    def stop_camera(self) -> None:
        self.camera.stop()
        self.camera_active = False
        self.start_button.configure(state=tk.NORMAL if self.analyzer else tk.DISABLED)
        self.stop_button.configure(state=tk.DISABLED)
        self.people_text.set("Recognized: —")
        self.similarity_text.set("Similarity: —")
        if self.analyzer:
            self.status_text.set("Camera stopped.")
        self.preview.configure(image="", text="Start the camera to see the live preview")
        self.preview_image = None

    def open_registration(self) -> None:
        if self.analyzer is None:
            messagebox.showerror(
                "Recognition unavailable",
                "Recognition models are not available. See README.md for setup instructions.",
                parent=self.root,
            )
            return
        if not self.camera.running:
            self.start_camera()
            if not self.camera.running:
                return

        dialog = tk.Toplevel(self.root)
        dialog.title("Register a face")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="Person's name:").pack(anchor=tk.W)
        name_entry = ttk.Entry(body, width=36)
        name_entry.pack(fill=tk.X, pady=(4, 10))
        name_entry.focus_set()
        ttk.Label(
            body,
            text="Look toward the camera. Capture 3 clear samples with only one face in view.",
            wraplength=380,
        ).pack(anchor=tk.W, pady=(0, 10))
        count_text = tk.StringVar(value="Captured samples: 0 / 3")
        ttk.Label(body, textvariable=count_text).pack(anchor=tk.W, pady=(0, 8))
        dialog_status = tk.StringVar(value="Capture samples from the live camera.")
        ttk.Label(body, textvariable=dialog_status, wraplength=380).pack(
            anchor=tk.W,
            pady=(0, 10),
        )

        samples: list[tuple[np.ndarray, np.ndarray]] = []

        def capture_sample() -> None:
            frame = self.camera.read()
            if frame is None:
                dialog_status.set("Waiting for a camera frame. Check the camera connection.")
                return
            try:
                faces = self.analyzer.extract(frame)
            except cv2.error as error:
                dialog_status.set(f"Face processing failed: {error}")
                return
            if len(faces) != 1:
                dialog_status.set(
                    "Show exactly one face in the camera view to capture a sample."
                )
                return
            sample = faces[0]
            samples.append((sample.aligned_face.copy(), sample.embedding.copy()))
            count_text.set(f"Captured samples: {len(samples)} / 3")
            dialog_status.set("Sample captured. Change your angle slightly for the next one.")
            if len(samples) >= 3:
                save_button.configure(state=tk.NORMAL)

        def save_registration() -> None:
            try:
                identity = self.registration.register(name_entry.get(), samples)
                self.identities = self.database.load_identities()
            except (ValueError, DataStoreError, OSError) as error:
                messagebox.showerror("Registration failed", str(error), parent=dialog)
                return
            self.status_text.set(f"Registered {identity.name}.")
            dialog.destroy()

        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Capture Sample", command=capture_sample).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        save_button = ttk.Button(
            buttons,
            text="Save Registration",
            command=save_registration,
            state=tk.DISABLED,
        )
        save_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)

    def _update_preview(self) -> None:
        if self.camera_active:
            frame = self.camera.read()
            if frame is not None:
                now = time.monotonic()
                if self.analyzer is not None and now - self.last_inference >= 0.15:
                    try:
                        self.recognitions = self.analyzer.recognize(frame, self.identities)
                    except cv2.error as error:
                        self.status_text.set(f"Face processing error: {error}")
                        self.recognitions = []
                    self.last_inference = now
                    self._update_recognition_labels()
                self._draw_recognitions(frame)
                self._show_frame(frame)
            elif not self.camera.running and self.camera.error:
                camera_error = self.camera.error
                self.stop_camera()
                self.status_text.set(camera_error)
        self.root.after(30, self._update_preview)

    def _draw_recognitions(self, frame: np.ndarray) -> None:
        for result in self.recognitions:
            x, y, width, height = result.box
            color = (54, 190, 90) if result.name != "UNKNOWN" else (50, 150, 230)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            label = result.name
            if result.similarity is not None:
                label += f" ({result.similarity:.2f})"
            cv2.putText(
                frame,
                label,
                (x, max(22, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

    def _update_recognition_labels(self) -> None:
        if not self.recognitions:
            self.status_text.set("No faces detected.")
            self.people_text.set("Recognized: —")
            self.similarity_text.set("Similarity: —")
            return
        names = [result.name for result in self.recognitions]
        self.status_text.set(f"Detected {len(names)} face(s).")
        self.people_text.set("Recognized: " + ", ".join(names))
        scores = [
            f"{result.name}: {result.similarity:.3f}"
            for result in self.recognitions
            if result.similarity is not None
        ]
        self.similarity_text.set(
            "Cosine similarity: " + ("; ".join(scores) if scores else "—")
        )

    def _show_frame(self, frame: np.ndarray) -> None:
        preview_width = max(320, self.preview.winfo_width())
        preview_height = max(240, self.preview.winfo_height())
        height, width = frame.shape[:2]
        scale = min(preview_width / width, preview_height / height)
        resized = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.preview_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=self.preview_image, text="")

    def close(self) -> None:
        self.camera.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
