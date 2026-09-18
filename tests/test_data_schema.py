import json
import tempfile
import unittest
from pathlib import Path

from bonehub_data_schema import (
    BoneHubDatasetIO,
    BoneLabelMap,
    SubjectInfo,
    DatasetInfo,
    bonehub_to_snomed,
    is_compatible_schema_version,
    __version__,
)


class TestSubjectInfo(unittest.TestCase):
    def test_subject_info_validation(self):
        # Test valid SubjectInfo creation
        subject_info = SubjectInfo(
            dataset_id=1,
            subject_id=1,
            source_subject_path="path/to/subject",
            age=30,
            image=False,
            gender="F",
            segmentation={"FEMUR_LEFT": 30},
        )
        self.assertEqual(subject_info.gender, "F")
        with self.assertRaises(ValueError):
            subject_info.set_segmentation_value("NOT_A_BONE", 30)
        with self.assertRaises(ValueError):
            subject_info.gender = "male"
        with self.assertRaises(ValueError):
            subject_info.segmentation = {"FEMUR_LEFT": 55}  # origin 5 and review 5 do not exist


class TestDatasetInfo(unittest.TestCase):
    def test_dataset_info_validation(self):
        # Test valid DatasetInfo creation
        dataset_info = DatasetInfo(
            dataset_id=1,
            name="Test Dataset",
            url="http://example.com/dataset",
            paper="http://example.com/paper",
            country="Netherlands",
            release_date="2024-01-01",
            version="1.0",
            remarks="This is a test dataset.",
            medical_image_included=True,
            license="CC BY 4.0",
        )
        self.assertEqual(dataset_info.name, "Test Dataset")
        with self.assertRaises(ValueError):
            dataset_info.dataset_id = "1"
        with self.assertRaises(ValueError):
            dataset_info.medical_image_included = "yes"  # Invalid value, should be a boolean


class TestSchemaVersion(unittest.TestCase):
    def _write_dataset(self, root: Path, schema_version):
        folder = root / "Dataset_001"
        folder.mkdir(parents=True)
        info = {"dataset_id": 1, "name": "Test Dataset"}
        if schema_version is not None:
            info["schema_version"] = schema_version
        (folder / "Dataset_info_001.json").write_text(json.dumps(info))
        (folder / "Subject_info_001.json").write_text(json.dumps([{"image": False, "segmentation": {"FEMUR_LEFT": 30}}]))

    def test_dataset_from_another_schema_is_refused(self):
        for old in (None, "0.1.0"):
            with self.subTest(schema_version=old), tempfile.TemporaryDirectory() as tmp:
                self._write_dataset(Path(tmp), old)
                with self.assertRaises(ValueError):
                    BoneHubDatasetIO(Path(tmp), 1)

    def test_dataset_from_this_schema_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_dataset(Path(tmp), __version__)
            self.assertEqual(len(BoneHubDatasetIO(Path(tmp), 1)), 1)
        self.assertTrue(is_compatible_schema_version(__version__.rsplit(".", 1)[0] + ".99"))  # patch releases match


if __name__ == "__main__":
    unittest.main()
