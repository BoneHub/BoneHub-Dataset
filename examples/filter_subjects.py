from pathlib import Path

from bonehub_data_schema import BoneHubDatasetIO


def collect_subjects(dataset_root: Path):
    subjects = []

    for dataset_id in dataset_root.glob("*"):
        if dataset_id.is_dir() and dataset_id.name.startswith("Dataset_"):
            dataset_id = int(dataset_id.name.split("_")[1])
            dataset_io = BoneHubDatasetIO(dataset_root, dataset_id)
            subjects.extend(dataset_io.subject_info)

    return subjects


if __name__ == "__main__":
    dataset_root = Path("Z:/BoneHub/BoneHub_Dataset")
    subjects = collect_subjects(dataset_root)
    
    # find subjects that have segmentation of L4 or L5 lumbar vertebra and tibia
    filtered_subjects = []
    for subject in subjects:
        if subject.segmentation:
            segmentations = "; ".join(subject.segmentation)
            if (("L4" in segmentations) or ("L5" in segmentations)) and ("TIBIA" in segmentations):
                filtered_subjects.append(subject)
    print(f"Found {len(filtered_subjects)} subjects with both L4 or L5 lumbar vertebra and TIBIA segmentation.")
    for subject in filtered_subjects:
        print(f"Dataset ID: {subject.dataset_id}, Subject ID: {subject.subject_id}")
