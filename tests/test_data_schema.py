import unittest

from bonehub_data_schema import BoneLabelMap, SubjectInfo, DatasetInfo, bonehub_to_snomed


class TestSubjectInfo(unittest.TestCase):
    def test_subject_info_validation(self):
        # Test valid SubjectInfo creation
        subject_info = SubjectInfo(
            dataset_id=1,
            subject_id=1,
            source_subject_path="path/to/subject",
            age=30,
            segmentation={"FEMUR_LEFT": 1},
        )
        with self.assertRaises(ValueError):
            subject_info.set_segmentation_value("FEMUR", 1)
        with self.assertRaises(ValueError):
            subject_info.segmentation = {"FEMUR_LEFT": 3}


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


if __name__ == "__main__":
    unittest.main()
