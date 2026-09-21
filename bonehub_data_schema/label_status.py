"""Status of a label's segmentation, mesh or NURBS file.

A label missing from a subject's `segmentation`, `mesh` or `nurbs` entry has status 0.
"""

VALID_LABEL_VALUES = {
    0: "not available",
    1: "available, not reviewed or corrected",
    2: "available, reviewed and corrected (if necessary)",
}


def check_label_status(value) -> None:
    # exact type check: True and 1.0 compare equal to 1
    if type(value) is not int or value not in VALID_LABEL_VALUES:
        raise ValueError(f"Invalid label status: {value!r}. Valid values are {VALID_LABEL_VALUES}")
