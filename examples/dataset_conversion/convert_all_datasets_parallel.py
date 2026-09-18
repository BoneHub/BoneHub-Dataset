"""Convert the BoneHub public datasets listed in DATASET_JOBS, several at a time.

Run from the repository root with the package installed (pip install -e ".[converter]")
and the MAX_SUBJECTS_FOR_TESTING environment variable unset; see --help for all options.
Each dataset writes a log into its Dataset_XXX folder.

To regenerate segmentations, meshes and subject info but keep the exported images:
1. In each Dataset_XXX folder keep Image/ and Subject_info_XXX.json, and delete
   Segmentation/, Mesh/ and NURBS/.
2. Convert one dataset and check its masks in 3D Slicer:

       python examples/dataset_conversion/convert_all_datasets_parallel.py --output-root Z:/BoneHub/BoneHub_Dataset --skip-existing-images --datasets vsd_reconstruction

3. Convert everything (sized for 32 GB of RAM):

       python examples/dataset_conversion/convert_all_datasets_parallel.py --output-root Z:/BoneHub/BoneHub_Dataset --skip-existing-images --max-workers 3 --subject-workers 4

4. Re-run examples/nurbs_conversion; NURBS surfaces are not produced here.
"""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
) -> str:
    dataset = job.dataset_class(job.source_root)
    dataset.export_to_bonehub_format(
        output_root=output_root,
        output_dataset_id=job.dataset_id,
        overwrite=overwrite or skip_existing_images,
        num_workers=subject_workers,
        verbose=verbose,
        skip_existing_images=skip_existing_images,
    )
    return f"{job.name} -> Dataset_{job.dataset_id:03d}"


def main() -> int:
    args = parse_args()
    jobs = select_jobs(args.datasets)

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    if not jobs:
        raise ValueError("No datasets selected for conversion.")

    output_root = args.output_root.resolve()
    print(f"Output root: {output_root}")
    print(f"Datasets queued: {', '.join(job.name for job in jobs)}")
    print(f"Converting up to {min(args.max_workers, len(jobs))} datasets at once")

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
            ): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
                print(f"[OK] {result}")
            except Exception as exc:
                failures.append((job, exc))
                print(f"[FAILED] {job.name}: {exc}")

    if failures:
        print("\nOne or more dataset conversions failed:")
        for job, exc in failures:
            print(f"- {job.name} (Dataset_{job.dataset_id:03d}): {exc}")
        return 1

    print("\nAll dataset conversions completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
