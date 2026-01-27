import unittest

import sys

sys.path.append(".")

from bonehub_segmentation.data_loaders import StandardSegmentationLoader, create_train_val_split, MSDLoader
from bonehub_segmentation.model_wrappers import (
    TotalSegmentatorWrapper,
    MOOSEWrapper,
    CustomNNUNetWrapper,
    UnifiedSegmentationInterface,
)


class ImportSmokeTests(unittest.TestCase):
    def test_data_loader_imports(self):
        self.assertIsNotNone(StandardSegmentationLoader)
        self.assertIsNotNone(create_train_val_split)
        self.assertIsNotNone(MSDLoader)

    def test_model_wrapper_imports(self):
        self.assertIsNotNone(TotalSegmentatorWrapper)
        self.assertIsNotNone(MOOSEWrapper)
        self.assertIsNotNone(CustomNNUNetWrapper)
        self.assertIsNotNone(UnifiedSegmentationInterface)


if __name__ == "__main__":
    unittest.main()
