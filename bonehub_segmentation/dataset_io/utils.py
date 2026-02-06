import os
from pathlib import Path
import pydicom


def get_dicom_subject_metadata(dicom_folder: str) -> dict:
    first_file = next(f for f in os.listdir(dicom_folder) if f.endswith(".dcm") or f.endswith(".dicom"))
    ds = pydicom.dcmread(os.path.join(dicom_folder, first_file), stop_before_pixels=True)
    age = getattr(ds, "PatientAge", None)
    if age:
        age = "".join(filter(str.isdigit, age))

    gender = getattr(ds, "PatientSex", None)
    if gender:
        gender = "male" if gender.lower() == "m" else "female" if gender.lower() == "f" else gender

    return {
        "age": age,
        "gender": gender,
    }


def create_train_val_split(
    image_dir: str,
    label_dir: str,
    val_split: float = 0.2,
    random_seed: int = 42,
) -> tuple:
    """Split dataset into training and validation sets.

    Args:
        image_dir: Path to directory containing images
        label_dir: Path to directory containing labels
        val_split: Fraction of data to use for validation (default: 0.2)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_images, train_labels, val_images, val_labels)
    """
    from sklearn.model_selection import train_test_split

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    # Get list of image files
    image_files = sorted([str(f) for f in image_dir.glob("*") if f.is_file()])

    # Get corresponding label files
    label_files = []
    for img_file in image_files:
        img_name = Path(img_file).stem
        label_file = list(label_dir.glob(f"{img_name}*"))
        if label_file:
            label_files.append(str(label_file[0]))
        else:
            label_files.append(None)

    # Split data
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        image_files,
        label_files,
        test_size=val_split,
        random_state=random_seed,
    )

    return list(train_imgs), list(train_labels), list(val_imgs), list(val_labels)
