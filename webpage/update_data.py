"""Generate webpage/data.js from BoneHub Dataset_info and Subject_info files.

The script scans all ``Dataset_info_*.json`` and ``Subject_info_*.json`` files under a configured root
directory and writes a JavaScript array consumed by the static webpage.
"""

import json
import os
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path

from bonehub_data_schema import SubjectInfo, BoneHubDatasetIO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBPAGE_DIR = PROJECT_ROOT / "webpage"
CONFIG_PATH = WEBPAGE_DIR / "config.ini"
OUTPUT_PATH = WEBPAGE_DIR / "data.js"
SUBJECT_INFO_FIELDS = tuple(SubjectInfo.model_fields.keys())

SUBJECT_KEY_MAPPINGS = {
    "dataset_id": "Dataset ID",
    "subject_id": "Subject ID",
    "age": "Age (years)",
    "gender": "Gender",
    "weight": "Weight (kg)",
    "height": "Height (cm)",
    "bmi": "BMI",
    "imaging_modality": "Imaging Modality",
    "image": "Image Available",
    "segmentation": "Segmentation Available",
    "mesh": "Mesh Available",
    "nurbs": "NURBS Available",
}

DATASET_KEY_MAPPINGS = {
    "dataset_id": "Dataset ID",
    "name": "Dataset Name",
    "url": "URL",
    "paper": "Paper",
    "license": "License",
}


def _format_structures(structures: dict | None) -> str:
    if not structures:
        return ""
    return "; ".join(f"{label}" for label, value in sorted(structures.items()))


def _load_configured_root() -> Path:
    config = ConfigParser()
    config.read(CONFIG_PATH)

    env_root = os.environ.get("BONEHUB_DATA_ROOT")
    if env_root:
        candidate = Path(env_root)
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()

    if config.has_option("BoneHub-Dataset", "root_dir"):
        root_dir = Path(config["BoneHub-Dataset"]["root_dir"])
    else:
        raise ValueError(f"Missing 'root_dir' in config file: {CONFIG_PATH}")

    return root_dir if root_dir.is_absolute() else (PROJECT_ROOT / root_dir).resolve()


def _collect_data(dataset_root: Path) -> tuple[list[dict], list[dict]]:
    subject_rows: list[dict] = []
    dataset_rows: list[dict] = []

    for dataset_id in dataset_root.glob("*"):
        if dataset_id.is_dir() and dataset_id.name.startswith("Dataset_"):
            dataset_id = int(dataset_id.name.split("_")[1])
            dataset_io = BoneHubDatasetIO(dataset_root, dataset_id)
            dataset_info = dataset_io.dataset_info
            subject_info = dataset_io.subject_info

            dataset_row = {new_key: getattr(dataset_info, old_key) for old_key, new_key in DATASET_KEY_MAPPINGS.items()}
            dataset_rows.append(dataset_row)

            for subject in subject_info:
                subject_row = {new_key: getattr(subject, old_key) for old_key, new_key in SUBJECT_KEY_MAPPINGS.items()}
                subject_row["Segmentation Available"] = _format_structures(subject.segmentation)
                subject_row["Mesh Available"] = _format_structures(subject.mesh)
                subject_row["NURBS Available"] = _format_structures(subject.nurbs)
                subject_rows.append(subject_row)

    return dataset_rows, subject_rows


def _write_data_js(dataset_rows: list[dict], subject_rows: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subject_payload = json.dumps(subject_rows, indent=2, ensure_ascii=False)
    dataset_payload = json.dumps(dataset_rows, indent=2, ensure_ascii=False)
    content = (
        "// Auto-generated from Dataset_info and Subject_info JSON files\n"
        "// Do not edit manually - run webpage/update_data.py instead\n"
        f"// Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"const datasetsData = {dataset_payload};\n"
        f"const subjectsData = {subject_payload};\n"
    )
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(content)


def main() -> int:
    dataset_root = _load_configured_root()
    dataset_rows, subject_rows = _collect_data(dataset_root)
    _write_data_js(dataset_rows, subject_rows)

    print(f"Generated {OUTPUT_PATH}")
    print(f"Dataset root: {dataset_root}")
    print(f"Subjects exported: {len(subject_rows)}")
    print(f"Datasets exported: {len(dataset_rows)}")
    return 0


if __name__ == "__main__":
    main()
