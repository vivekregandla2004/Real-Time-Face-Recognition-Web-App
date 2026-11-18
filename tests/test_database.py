import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.database import DataStoreError, FaceDatabase


class FaceDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database = FaceDatabase(
            registry_path=self.root / "registry.json",
            dataset_dir=self.root / "registered_faces",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_registration_saves_face_image_metadata_and_embedding(self):
        image = np.full((112, 112, 3), 127, dtype=np.uint8)
        embedding = np.zeros(128, dtype=np.float32)
        embedding[0] = 1.0

        identity = self.database.register("  Ada Lovelace  ", [(image, embedding)])

        self.assertEqual(identity.name, "Ada Lovelace")
        self.assertEqual(len(identity.image_paths), 1)
        self.assertTrue((self.root / "registered_faces" / identity.image_paths[0]).is_file())
        saved = json.loads((self.root / "registry.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["identities"][0]["name"], "Ada Lovelace")
        self.assertEqual(len(saved["embeddings"][identity.identity_id]), 1)

    def test_loading_registered_identity_restores_embedding(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        embedding = np.ones(128, dtype=np.float32)
        self.database.register("Grace", [(image, embedding)])

        loaded = self.database.load_identities()

        self.assertEqual([identity.name for identity in loaded], ["Grace"])
        np.testing.assert_allclose(loaded[0].embeddings[0], embedding)

    def test_rejects_empty_name_and_corrupted_registry(self):
        with self.assertRaises(ValueError):
            self.database.register("  ", [])
        self.database.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.database.registry_path.write_text("{", encoding="utf-8")

        with self.assertRaises(DataStoreError):
            self.database.load_identities()


if __name__ == "__main__":
    unittest.main()
