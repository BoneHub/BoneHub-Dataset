import unittest

from bonehub_data_schema import BoneLabelMap, SubjectInfo, DatasetInfo, bonehub_to_snomed


class ImportSmokeTests(unittest.TestCase):
    def test_imports(self):
        # Test that we can import the classes and functions without errors
        dataset_info = DatasetInfo(dataset_id=1, name="Test Dataset")
        subject_info = SubjectInfo(dataset_id=1, subject_id=1, image=0, age=30)
        subject_info.set_segmentation_value(BoneLabelMap.FEMUR_BOTH.name, 1)


if __name__ == "__main__":
    unittest.main()
