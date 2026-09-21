import unittest

from bonehub_data_schema import VALID_LABEL_VALUES, SubjectInfo


class TestLabelStatus(unittest.TestCase):
    def test_valid_values(self):
        self.assertEqual(sorted(VALID_LABEL_VALUES), [0, 1, 2])
        subject = SubjectInfo(image=True)
        for value in VALID_LABEL_VALUES:
            subject.set_segmentation_value("FEMUR_LEFT", value)

    def test_invalid_values_are_rejected(self):
        subject = SubjectInfo(image=True)
        for bad in (3, -1, 10, 30, True, "1", None, 1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                subject.set_segmentation_value("FEMUR_LEFT", bad)

    def test_available_labels_skip_status_zero_and_missing_labels(self):
        subject = SubjectInfo(image=True)
        for setter in (subject.set_segmentation_value, subject.set_mesh_value, subject.set_nurbs_value):
            setter("TIBIA_LEFT", 2)
            setter("FEMUR_LEFT", 1)
            setter("FEMUR_RIGHT", 0)
        for kind in ("segmentation", "mesh", "nurbs"):
            with self.subTest(kind=kind):
                self.assertEqual(subject.available_labels(kind), ["FEMUR_LEFT", "TIBIA_LEFT"])


if __name__ == "__main__":
    unittest.main()
