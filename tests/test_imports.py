import unittest

import sys

sys.path.append(".")

from bonehub_segmentation.dataset_io import BaseDatasetReader, BoneHubDatasetReader, export_custom_dataset_to_bonehub_format
from bonehub_segmentation.model_wrappers import (
    TotalSegmentatorWrapper,
    MOOSEWrapper,
    CustomNNUNetWrapper,
    UnifiedSegmentationInterface,
)


class ImportSmokeTests(unittest.TestCase):
    def test_data_loader_imports(self):
        self.assertIsNotNone(BaseDatasetReader)
        self.assertIsNotNone(BoneHubDatasetReader)
        self.assertIsNotNone(export_custom_dataset_to_bonehub_format)

    def test_model_wrapper_imports(self):
        self.assertIsNotNone(TotalSegmentatorWrapper)
        self.assertIsNotNone(MOOSEWrapper)
        self.assertIsNotNone(CustomNNUNetWrapper)
        self.assertIsNotNone(UnifiedSegmentationInterface)


if __name__ == "__main__":
    unittest.main()
