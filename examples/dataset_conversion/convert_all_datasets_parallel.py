"""Convert the BoneHub public datasets listed in DATASET_JOBS, several at a time.

Run from the repository root with the package installed (pip install -e ".[converter]")
and the MAX_SUBJECTS_FOR_TESTING environment variable unset; see --help for all options.
Each dataset writes a log into its Dataset_XXX folder, naming any subject that failed with its traceback;
the other subjects are still converted. The run's own log (Conversion_*.log in the output root) ends with a
summary of which datasets succeeded and which failed.

To regenerate segmentations, meshes and subject info but keep the exported images:
1. In each Dataset_XXX folder keep Image/ and Subject_info_XXX.json, and delete
   Segmentation/, Mesh/ and NURBS/.
2. Convert one dataset and check its masks in 3D Slicer:

       python examples/dataset_conversion/convert_all_datasets_parallel.py --output-root Z:/BoneHub/BoneHub_Dataset --skip-existing-images --datasets vsd_reconstruction

3. Convert everything (sized for 32 GB of RAM):

       python examples/dataset_conversion/convert_all_datasets_parallel.py --output-root Z:/BoneHub/BoneHub_Dataset --skip-existing-images --max-workers 3 --subject-workers 4

    To retry only the subjects that failed (or were not reached) in an earlier run:

       python examples/dataset_conversion/convert_all_datasets_parallel.py --output-root Z:/BoneHub/BoneHub_Dataset --skip-existing-subjects --datasets vsd_reconstruction
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import bonehub_dataset_converter.custom_dataset_io as custom_dataset_io


@dataclass(frozen=True)
class DatasetJob:
    name: str
    dataset_id: int
    dataset_class: type
    source_root: Path


DATASET_JOBS: tuple[DatasetJob, ...] = (
    DatasetJob(
        name="bonedat",
        dataset_id=1,
        dataset_class=custom_dataset_io.BoneDat,
        source_root=Path("Z:/BoneHub/Public_Datasets/124 BoneDat/BoneDat"),
    ),
    DatasetJob(
        name="kits2023",
        dataset_id=2,
        dataset_class=custom_dataset_io.KiTS2023,
        source_root=Path("Z:/BoneHub/Public_Datasets/073 kits23"),
    ),
    DatasetJob(
        name="spine_mets_ct_seg",
        dataset_id=3,
        dataset_class=custom_dataset_io.SpineMetsCTSeg,
        source_root=Path("Z:/BoneHub/Public_Datasets/019 TCIA Spine-Mets-CT-SEG"),
    ),
    DatasetJob(
        name="enhance_pet",
        dataset_id=4,
        dataset_class=custom_dataset_io.EnhancePET,
        source_root=Path("Z:/BoneHub/Public_Datasets/134 enhance-pet-1_6k"),
    ),
    DatasetJob(
        name="vsd_reconstruction",
        dataset_id=5,
        dataset_class=custom_dataset_io.VSDReconstruction,
        source_root=Path("Z:/BoneHub/Public_Datasets/036 VSDFullBodyBoneReconstruction/Hamid_processed/vsd-lower-extremities-seg"),
    ),
    DatasetJob(
        name="totalsegmentator_ct",
        dataset_id=6,
        dataset_class=custom_dataset_io.TotalSegmentatorCT,
        source_root=Path("Z:/BoneHub/Public_Datasets/027 Totalsegmentator/raw_data"),
    ),
    DatasetJob(
        name="ctpelvic1k",
        dataset_id=7,
        dataset_class=custom_dataset_io.CTPelvic1K,
        source_root=Path("Z:/BoneHub/Public_Datasets/044 CTPelvic1K"),
    ),
    DatasetJob(
        name="pengwin",
        dataset_id=8,
        dataset_class=custom_dataset_io.PENGWIN,
        source_root=Path("Z:/BoneHub/Public_Datasets/106 PENGWIN"),
    ),
    DatasetJob(
        name="ctpel",
        dataset_id=9,
        dataset_class=custom_dataset_io.CTPEL,
        source_root=Path("Z:/BoneHub/Public_Datasets/138 ctpel"),
    ),
    DatasetJob(
        name="synthrad2023",
        dataset_id=10,
        dataset_class=custom_dataset_io.SynthRAD2023,
        source_root=Path("Z:/BoneHub/Public_Datasets/006 SynthRAD2023"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BoneHub public datasets in parallel.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root folder where converted BoneHub datasets will be written.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=len(DATASET_JOBS),
        help="Maximum number of datasets converted at once, each in its own process.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[job.name for job in DATASET_JOBS],
        help="Optional subset of dataset names to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output dataset folders if they already exist.",
    )
    parser.add_argument(
        "--subject-workers",
        type=int,
        default=None,
        help="Subjects converted at once within each dataset, each in its own process "
        "(default: one per CPU core, at most 8). Up to max-workers x subject-workers processes run "
        "together, each needing up to ~2 GB for the largest masks; lower it if memory runs out.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide the per-dataset progress bars.",
    )
    parser.add_argument(
        "--skip-existing-images",
        action="store_true",
        help="Keep exported images whose source subject is unchanged and regenerate everything else. "
        "Implies --overwrite.",
    )
    parser.add_argument(
        "--skip-existing-subjects",
        action="store_true",
        help="Convert only the subjects missing from Subject_info_XXX.json (e.g. those that failed or were not "
        "reached in an earlier run) and keep the rest. Implies --overwrite.",
    )
    return parser.parse_args()


def select_jobs(selected_names: Sequence[str] | None) -> list[DatasetJob]:
    if not selected_names:
        return list(DATASET_JOBS)
    selected = set(selected_names)
    return [job for job in DATASET_JOBS if job.name in selected]


def convert_dataset(
    job: DatasetJob,
    output_root: Path,
    overwrite: bool,
    verbose: bool,
    skip_existing_images: bool = False,
    subject_workers: int | None = None,
    skip_existing_subjects: bool = False,
) -> Path:
    dataset = job.dataset_class(job.source_root)
    return dataset.export_to_bonehub_format(
        output_root=output_root,
        output_dataset_id=job.dataset_id,
        overwrite=overwrite or skip_existing_images or skip_existing_subjects,
        num_workers=subject_workers,
        skip_existing_subjects=skip_existing_subjects,
        skip_existing_images=skip_existing_images,
        verbose=verbose,
    )


def setup_run_logger(output_root: Path) -> logging.Logger:
    """Log to the console and to a run log in output_root, next to the Dataset_XXX folders."""
    output_root.mkdir(parents=True, exist_ok=True)
    log_file_path = output_root / f"Conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger = logging.getLogger("convert_all_datasets")
    logger.setLevel(logging.INFO)
    logger.handlers = [logging.StreamHandler(), file_handler]
    logger.info(f"Run log: {log_file_path}")
    return logger


def log_summary(logger: logging.Logger, successes: list[tuple[DatasetJob, Path]], failures: list[tuple[DatasetJob, Exception]]) -> None:
    lines = [f"Conversion summary: {len(successes)} succeeded, {len(failures)} failed."]
    for job, log_file_path in sorted(successes, key=lambda item: item[0].dataset_id):
        lines.append(f"  [OK]     {job.name} (Dataset_{job.dataset_id:03d}), log: {log_file_path}")
    for job, exc in sorted(failures, key=lambda item: item[0].dataset_id):
        lines.append(f"  [FAILED] {job.name} (Dataset_{job.dataset_id:03d}): {exc}")
    if failures:
        lines.append(
            "Refer to each failed dataset's log file to see which subjects failed and why. Subjects that failed are "
            "null in Subject_info_XXX.json; the other subjects were converted and saved. "
            "Run again with --skip-existing-subjects to convert only the missing subjects."
        )
    logger.log(logging.ERROR if failures else logging.INFO, "\n".join(lines))


def main() -> int:
    args = parse_args()
    jobs = select_jobs(args.datasets)

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    if not jobs:
        raise ValueError("No datasets selected for conversion.")

    output_root = args.output_root.resolve()
    logger = setup_run_logger(output_root)
    logger.info(f"Output root: {output_root}")
    logger.info(f"Datasets queued: {', '.join(job.name for job in jobs)}")
    logger.info(f"Converting up to {min(args.max_workers, len(jobs))} datasets at once")

    successes: list[tuple[DatasetJob, Path]] = []
    failures: list[tuple[DatasetJob, Exception]] = []

    # One process per dataset, so a crash in one does not stop the others.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(jobs)), mp_context=context) as executor:
        future_to_job = {
            executor.submit(
                convert_dataset,
                job,
                output_root,
                args.overwrite,
                not args.quiet,
                args.skip_existing_images,
                args.subject_workers,
                args.skip_existing_subjects,
            ): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                successes.append((job, future.result()))
                logger.info(f"[OK] {job.name} -> Dataset_{job.dataset_id:03d}")
            except Exception as exc:
                failures.append((job, exc))
                logger.error(f"[FAILED] {job.name} -> Dataset_{job.dataset_id:03d}: {exc}")

    log_summary(logger, successes, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
