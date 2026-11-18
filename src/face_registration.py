from typing import Sequence

import numpy as np

from .database import FaceDatabase, Identity


class FaceRegistration:
    def __init__(self, database: FaceDatabase):
        self.database = database

    def register(
        self,
        name: str,
        samples: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> Identity:
        return self.database.register(name, samples)
