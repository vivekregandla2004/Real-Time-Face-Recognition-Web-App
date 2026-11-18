import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .utils import DATASET_DIR, REGISTRY_PATH


logger = logging.getLogger(__name__)


class DataStoreError(Exception):
    """Raised when registration data cannot be read or safely updated."""


@dataclass
class Identity:
    identity_id: str
    name: str
    image_paths: list[str]
    embeddings: list[np.ndarray]


class FaceDatabase:
    def __init__(self, registry_path: Path = REGISTRY_PATH, dataset_dir: Path = DATASET_DIR):
        self.registry_path = Path(registry_path)
        self.dataset_dir = Path(dataset_dir)

    def load_identities(self) -> list[Identity]:
        if not self.registry_path.exists():
            return []

        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DataStoreError(f"Could not read registration data: {error}") from error

        if not isinstance(data, dict) or data.get("version") != 1:
            raise DataStoreError("Registration data has an unsupported or invalid format.")

        identity_rows = data.get("identities")
        embedding_rows = data.get("embeddings")
        if not isinstance(identity_rows, list) or not isinstance(embedding_rows, dict):
            raise DataStoreError("Registration data is missing identity or embedding records.")

        identities: list[Identity] = []
        for row in identity_rows:
            if not isinstance(row, dict):
                logger.warning("Skipping malformed identity record.")
                continue
            identity_id = row.get("id")
            name = row.get("name")
            image_paths = row.get("images", [])
            raw_embeddings = embedding_rows.get(identity_id)
            if (
                not isinstance(identity_id, str)
                or not isinstance(name, str)
                or not name.strip()
                or not isinstance(image_paths, list)
                or not isinstance(raw_embeddings, list)
            ):
                logger.warning("Skipping incomplete identity record.")
                continue
            try:
                embeddings = [np.asarray(vector, dtype=np.float32) for vector in raw_embeddings]
                if not embeddings or any(vector.shape != (128,) for vector in embeddings):
                    raise ValueError("embedding must contain 128 values")
                if not all(np.isfinite(vector).all() for vector in embeddings):
                    raise ValueError("embedding contains non-finite values")
            except (TypeError, ValueError) as error:
                logger.warning("Skipping invalid embeddings for %s: %s", name, error)
                continue
            identities.append(
                Identity(
                    identity_id=identity_id,
                    name=name,
                    image_paths=[str(path) for path in image_paths],
                    embeddings=embeddings,
                )
            )
        return identities

    def register(
        self,
        name: str,
        samples: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> Identity:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Enter a person's name before registering.")
        if not samples:
            raise ValueError("Capture at least one clear face sample.")

        identity_id = uuid.uuid4().hex
        person_dir = self.dataset_dir / identity_id
        person_dir.mkdir(parents=True, exist_ok=False)
        saved_files: list[Path] = []
        relative_paths: list[str] = []
        embeddings: list[list[float]] = []

        try:
            for index, (image, embedding) in enumerate(samples, start=1):
                vector = np.asarray(embedding, dtype=np.float32)
                if vector.shape != (128,) or not np.isfinite(vector).all():
                    raise ValueError("A face sample has an invalid embedding.")
                success, encoded_image = cv2.imencode(".jpg", image)
                if not success:
                    raise DataStoreError("Could not encode a captured face image.")
                image_path = person_dir / f"sample_{index:02d}.jpg"
                image_path.write_bytes(encoded_image.tobytes())
                saved_files.append(image_path)
                relative_paths.append(str(image_path.relative_to(self.dataset_dir)))
                embeddings.append(vector.tolist())

            data = self._read_registry()
            data["identities"].append(
                {
                    "id": identity_id,
                    "name": clean_name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "images": relative_paths,
                }
            )
            data["embeddings"][identity_id] = embeddings
            self._write_registry(data)
        except Exception:
            for image_path in saved_files:
                image_path.unlink(missing_ok=True)
            person_dir.rmdir()
            raise

        return Identity(
            identity_id=identity_id,
            name=clean_name,
            image_paths=relative_paths,
            embeddings=[np.asarray(vector, dtype=np.float32) for vector in embeddings],
        )

    def _read_registry(self) -> dict:
        if not self.registry_path.exists():
            return {"version": 1, "identities": [], "embeddings": {}}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DataStoreError(f"Could not update registration data: {error}") from error
        if (
            not isinstance(data, dict)
            or data.get("version") != 1
            or not isinstance(data.get("identities"), list)
            or not isinstance(data.get("embeddings"), dict)
        ):
            raise DataStoreError("Registration data has an unsupported or invalid format.")
        return data

    def _write_registry(self, data: dict) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.registry_path.with_suffix(".json.tmp")
        try:
            temporary_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(self.registry_path)
        except OSError as error:
            temporary_path.unlink(missing_ok=True)
            raise DataStoreError(f"Could not save registration data: {error}") from error
