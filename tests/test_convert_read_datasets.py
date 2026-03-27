import os

os.environ["MAX_SUBJECTS_FOR_TESTING"] = "2"  # Set a limit for testing purposes

from pathlib import Path
import unittest

from bonehub_data_schema import BoneHubDatasetIO

import bonehub_dataset_converter.custom_dataset_io as custom_dataset_io


_BONEHUB_DATASET_ROOT = Path("./tests/dummy_bonehub_dataset")


# @unittest.skip("")
class TestBoneDat(unittest.TestCase):
    def test_convert(self):

        public_datasets_root = Path("Z:/BoneHub/Public_Datasets/124 BoneDat/BoneDat")
        public_dataset = custom_dataset_io.BoneDat(public_datasets_root)
        public_dataset.export_to_bonehub_format(output_root=_BONEHUB_DATASET_ROOT, output_dataset_id=1, overwrite=True)

    def test_read(self):
        dataset = BoneHubDatasetIO(_BONEHUB_DATASET_ROOT, dataset_id=1)

        self.assertEqual(dataset.dataset_info.dataset_id, 1)
        self.assertEqual(dataset.subject_info[0].subject_id, 1)
        self.assertEqual(len(dataset), 2)
        self.assertTrue(dataset.check_dataset_integrity())


# @unittest.skip("")
class TestKits2023(unittest.TestCase):
    def test_convert(self):
        public_datasets_root = Path("Z:/BoneHub/Public_Datasets/073 kits23")
        public_dataset = custom_dataset_io.KiTS2023(public_datasets_root)
        public_dataset.export_to_bonehub_format(output_root=_BONEHUB_DATASET_ROOT, output_dataset_id=2, overwrite=True)

    def test_read(self):
        dataset = BoneHubDatasetIO(_BONEHUB_DATASET_ROOT, dataset_id=2)

        self.assertEqual(dataset.dataset_info.dataset_id, 2)
        self.assertEqual(dataset.subject_info[0].subject_id, 1)
        self.assertEqual(len(dataset), 2)
        self.assertTrue(dataset.check_dataset_integrity())


# @unittest.skip("")
class TestSpineMetsCTSeg(unittest.TestCase):
    def test_convert(self):
        public_datasets_root = Path("Z:/BoneHub/Public_Datasets/019 TCIA Spine-Mets-CT-SEG")
        public_dataset = custom_dataset_io.SpineMetsCTSeg(public_datasets_root)
        public_dataset.export_to_bonehub_format(output_root=_BONEHUB_DATASET_ROOT, output_dataset_id=3, overwrite=True)

    def test_read(self):
        dataset = BoneHubDatasetIO(_BONEHUB_DATASET_ROOT, dataset_id=3)

        self.assertEqual(dataset.dataset_info.dataset_id, 3)
        self.assertEqual(dataset.subject_info[0].subject_id, 1)
        self.assertEqual(len(dataset), 2)
        self.assertTrue(dataset.check_dataset_integrity())


# @unittest.skip("")
class TestACRIN6664(unittest.TestCase):
    def test_convert(self):
        public_datasets_root = Path("Z:/BoneHub/Public_Datasets/070 TCIA CT COLONOGRAPHY ACRIN 6664")
        public_dataset = custom_dataset_io.ACRIN6664(public_datasets_root)
        public_dataset.export_to_bonehub_format(output_root=_BONEHUB_DATASET_ROOT, output_dataset_id=4, overwrite=True)

    def test_read(self):
        dataset = BoneHubDatasetIO(_BONEHUB_DATASET_ROOT, dataset_id=4)

        self.assertEqual(dataset.dataset_info.dataset_id, 4)
        self.assertEqual(dataset.subject_info[0].subject_id, 1)
        self.assertEqual(len(dataset), 4)
        # The ACRIN 6664 dataset contains 4 valid subjects after applying the file count filter, even with the MAX_SUBJECTS_FOR_TESTING limit set to 2.
        # This is because the limit is applied to the number of subjects processed, but each subject may contain multiple valid subdirectories that are included in the dataset.
        # Therefore, we expect to have 4 valid subjects in total after processing the dataset, which is reflected in this assertion.
        self.assertTrue(dataset.check_dataset_integrity())


# @unittest.skip("")
class TestVSDReconstruction(unittest.TestCase):
    def test_convert(self):
        data_root = Path("Z:/BoneHub/Public_Datasets/036 VSDFullBodyBoneReconstruction/Hamid_processed")
        public_dataset = custom_dataset_io.VSDReconstruction(data_root)
        public_dataset.export_to_bonehub_format(output_root=_BONEHUB_DATASET_ROOT, output_dataset_id=5, overwrite=True)

    def test_read(self):
        dataset = BoneHubDatasetIO(_BONEHUB_DATASET_ROOT, dataset_id=5)

        self.assertEqual(dataset.dataset_info.dataset_id, 5)
        self.assertEqual(dataset.subject_info[0].subject_id, 1)
        self.assertEqual(len(dataset), 2)
        self.assertTrue(dataset.check_dataset_integrity())


# @unittest.skip("")
class TestEnhancePET(unittest.TestCase):
    def test_convert(self):
        data_root = Path("Z:/BoneHub/Public_Datasets/134 enhance-pet-1_6k")
        public_dataset = custom_dataset_io.EnhancePET(data_root)
        public_dataset.export_to_bonehub_format(output_root=_BONEHUB_DATASET_ROOT, output_dataset_id=6, overwrite=True)

    def test_read(self):
        dataset = BoneHubDatasetIO(_BONEHUB_DATASET_ROOT, dataset_id=6)

        self.assertEqual(dataset.dataset_info.dataset_id, 6)
        self.assertEqual(dataset.subject_info[0].subject_id, 1)
        self.assertEqual(len(dataset), 2)
        self.assertTrue(dataset.check_dataset_integrity())


# @unittest.skip("")
class TestTotalSegmentatorCT(unittest.TestCase):
    def test_convert(self):
        data_root = Path("Z:/BoneHub/Public_Datasets/027 Totalsegmentator/raw_data")
        public_dataset = custom_dataset_io.TotalSegmentatorCT(data_root)
        public_dataset.export_to_bonehub_format(output_root=_BONEHUB_DATASET_ROOT, output_dataset_id=7, overwrite=True)

    def test_read(self):
        dataset = BoneHubDatasetIO(_BONEHUB_DATASET_ROOT, dataset_id=7)

        self.assertEqual(dataset.dataset_info.dataset_id, 7)
        self.assertEqual(dataset.subject_info[0].subject_id, 1)
        self.assertEqual(len(dataset), 2)
        self.assertTrue(dataset.check_dataset_integrity())


if __name__ == "__main__":
    unittest.main()
