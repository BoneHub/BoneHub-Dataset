import importlib.util
import tempfile
import unittest
from pathlib import Path

from bonehub_data_schema import (
    BoneLabelMap,
    LabelStatus,
    Origin,
    Review,
    read_segmentation_labels,
    write_segmentation,
)

HAS_IO = all(importlib.util.find_spec(m) is not None for m in ("numpy", "SimpleITK"))


@unittest.skipUnless(HAS_IO, 'needs the io extra: pip install "bonehub-dataset[io]"')
class TestSegmentationFile(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import SimpleITK as sitk

        self.np, self.sitk = np, sitk
        self.tmp = Path(tempfile.mkdtemp())
        self.array = np.zeros((4, 8, 8), dtype=np.int32)
        self.array[0:2] = BoneLabelMap.FEMUR_LEFT.value
        self.array[2:4, :4] = BoneLabelMap.FEMUR_PROXIMAL_CORTICAL_LEFT.value
        self.reference = sitk.GetImageFromArray(np.zeros(self.array.shape, dtype=np.int16))

    def test_round_trip_keeps_voxels_names_and_statuses(self):
        status = {
            BoneLabelMap.FEMUR_LEFT.value: LabelStatus.of(Origin.SOURCE, Review.EXPERT_ACCEPTED),
            BoneLabelMap.FEMUR_PROXIMAL_CORTICAL_LEFT.value: LabelStatus.of(Origin.BONEHUB_AUTOMATIC),
        }
        written = write_segmentation(self.array, self.reference, self.tmp / "001_000001.seg.nrrd", status)
        path = self.tmp / "001_000001.seg.nrrd"

        self.assertEqual(read_segmentation_labels(path), written)
        self.assertEqual(written, {"FEMUR_LEFT": 33, "FEMUR_PROXIMAL_CORTICAL_LEFT": 10})
        voxels = self.sitk.GetArrayFromImage(self.sitk.ReadImage(str(path)))
        self.assertTrue(self.np.array_equal(voxels, self.array))

    def test_wrong_suffix_is_rejected(self):
        with self.assertRaises(ValueError):
            write_segmentation(self.array, self.reference, self.tmp / "001_000002.nii.gz", LabelStatus.of(Origin.SOURCE))
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_segment_without_a_status_tag_reads_as_unknown(self):
        image = self.sitk.GetImageFromArray(self.array)
        image.SetMetaData("Segment0_Name", "FEMUR_LEFT")  # e.g. a segment added by hand in 3D Slicer
        path = self.tmp / "manual.seg.nrrd"
        self.sitk.WriteImage(image, str(path))
        self.assertEqual(read_segmentation_labels(path), {"FEMUR_LEFT": None})


if __name__ == "__main__":
    unittest.main()
