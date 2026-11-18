import unittest

import numpy as np

from src.database import Identity
from src.face_recognition import match_embedding


class RecognitionMatchingTests(unittest.TestCase):
    def setUp(self):
        self.known_vector = np.zeros(128, dtype=np.float32)
        self.known_vector[0] = 1.0
        self.identity = Identity(
            identity_id="person-1",
            name="Ada",
            image_paths=[],
            embeddings=[self.known_vector],
        )

    def test_matches_registered_identity_by_cosine_similarity(self):
        query = self.known_vector * 2

        name, similarity = match_embedding(query, [self.identity])

        self.assertEqual(name, "Ada")
        self.assertAlmostEqual(similarity, 1.0)

    def test_returns_unknown_for_a_dissimilar_embedding(self):
        query = np.zeros(128, dtype=np.float32)
        query[1] = 1.0

        name, similarity = match_embedding(query, [self.identity])

        self.assertIsNone(name)
        self.assertAlmostEqual(similarity, 0.0)

    def test_returns_unknown_when_gallery_is_empty(self):
        name, similarity = match_embedding(self.known_vector, [])

        self.assertIsNone(name)
        self.assertIsNone(similarity)


if __name__ == "__main__":
    unittest.main()
