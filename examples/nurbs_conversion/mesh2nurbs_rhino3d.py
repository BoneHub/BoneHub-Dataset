"""Convert the meshes of BoneHub datasets to NURBS surfaces (IGES format) using Rhino3D.

For every subject of each given dataset, each mesh in Mesh/<subject>/ is converted with the
`mesh2nurbs-rhino3d` CLI and saved to NURBS/<subject>/, and the subject info is updated with
the new NURBS surfaces. The quad remesh edge length is chosen per bone from its body region
(see QUADREMESH_LENGTH_PER_REGION and QUADREMESH_LENGTH_PER_LABEL below).

Requires the `mesh2nurbs-rhino3d` CLI on PATH, e.g. by activating its conda env first:
    conda activate mesh2nurbs-rhino3d

Usage:
    python mesh2nurbs_rhino3d.py 5                      # convert all meshes of dataset 5 (VSD)
    python mesh2nurbs_rhino3d.py 5 7 9                  # several datasets
    python mesh2nurbs_rhino3d.py 5 --subjects 1 2       # only some subjects
    python mesh2nurbs_rhino3d.py 5 --overwrite          # regenerate existing NURBS surfaces
    python mesh2nurbs_rhino3d.py 5 --dry-run            # print the commands without running them
    python mesh2nurbs_rhino3d.py 5 --dataset-root "Z:/BoneHub/BoneHub_Dataset" --rhino-path "C:/Program Files/Rhino 8/System/Rhino.exe"

Run `python mesh2nurbs_rhino3d.py --help` for all options.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from bonehub_data_schema import BoneHubDatasetIO, BoneLabelMap
from bonehub_data_schema.labelmap import structure_of

DEFAULT_DATASET_ROOT = Path("Z:/BoneHub/BoneHub_Dataset")

CLI_COMMAND = "mesh2nurbs-rhino3d"
DEFAULT_PREPROCESSING_STEPS = ["remove-isolated-islands"]

# Quad remesh edge length (mm) per body region, the first digit of the label's structure id
# (see BoneLabelMap): 1 head, 2 spine, 3 thorax, 4 shoulder+arm, 5 hand, 6 pelvis, 7 leg, 8 foot.
QUADREMESH_LENGTH_PER_REGION = {
    1: 3.0,
    2: 2.0,
    3: 2.0,
    4: 3.0,
    5: 1.0,
    6: 3.0,
    7: 3.0,
    8: 1.0,
}
# Bones that need a different edge length than the rest of their region
QUADREMESH_LENGTH_PER_LABEL = {"PATELLA_LEFT": 2.0, "PATELLA_RIGHT": 2.0}
DEFAULT_QUADREMESH_LENGTH = 2.0


def quadremesh_length(label: str) -> float:
    if label in QUADREMESH_LENGTH_PER_LABEL:
        return QUADREMESH_LENGTH_PER_LABEL[label]
    region = structure_of(BoneLabelMap[label].value) // 1000
    return QUADREMESH_LENGTH_PER_REGION.get(region, DEFAULT_QUADREMESH_LENGTH)


def build_command(input_path: Path, label: str, preprocessing_steps: list, rhino_path: str | None) -> list:
    cmd = [
        CLI_COMMAND,
        "--input-path",
        str(input_path),
        "--quadremesh-length",
        str(quadremesh_length(label)),
        "--preprocessing-steps",
        *preprocessing_steps,
        "--output-filetype",
        "iges",
    ]
    if rhino_path:
        cmd += ["--rhino-path", rhino_path]
    return cmd


def convert_dataset(dataset_root: Path, dataset_id: int, args: argparse.Namespace) -> list:
    """Convert the meshes of one dataset and return the meshes that failed."""
    dataset = BoneHubDatasetIO(dataset_root, dataset_id)
    print(f"\n=== Dataset {dataset_id}: {dataset.dataset_path} ===", flush=True)

    failures = []
    for sub in dataset.subject_info:
        if args.subjects and sub.subject_id not in args.subjects:
            continue
        mesh_paths = dataset.get_mesh_paths(sub)
        if not mesh_paths:
            print(f"[skip] subject {sub.subject_id}: no meshes")
            continue
        for bonename, mesh_path in mesh_paths.items():
            copy_path = dataset.dataset_path / "NURBS" / mesh_path.parent.name / (mesh_path.stem + ".iges")
            if copy_path.exists() and not args.overwrite:
                print(f"NURBS surface already exists at `{copy_path}`. Skipping generation.")
                sub.set_nurbs_value(bonename, 1)
                continue

            # The CLI writes <stem>.iges next to the input mesh, so run it on a copy in a
            # temporary folder rather than writing into the dataset's Mesh folder.
            with tempfile.TemporaryDirectory() as tmp:
                input_path = Path(tmp) / mesh_path.name
                output_file_path = input_path.with_suffix(".iges")
                cmd = build_command(input_path, bonename, args.preprocessing_steps, args.rhino_path)
                print(f"Processing `{mesh_path}` ...")
                print(f"$ {subprocess.list2cmdline(cmd)}", flush=True)
                if args.dry_run:
                    continue

                shutil.copyfile(mesh_path, input_path)
                returncode = subprocess.run(cmd, cwd=tmp).returncode
                if returncode != 0 or not output_file_path.exists():
                    print(f"WARNING: Failed to create NURBS surface for `{mesh_path}` (exit code {returncode}).")
                    failures.append(mesh_path)
                    continue
                copy_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(output_file_path, copy_path)
                print(f"Successfully created NURBS surface at `{copy_path}`.")
                sub.set_nurbs_value(bonename, 1)

    if args.dry_run:
        return failures

    dataset.save_subject_info()
    print("Successfully updated the subject info with the new NURBS surface information.")
    if dataset.check_dataset_integrity():
        print("Dataset integrity check passed.")
    else:
        print("WARNING: Dataset integrity check failed. Please investigate the issues.")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_ids", nargs="+", type=int, help="IDs of the BoneHub datasets to convert")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Root folder of the BoneHub dataset (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to convert (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate NURBS surfaces that already exist")
    parser.add_argument(
        "--preprocessing-steps",
        nargs="+",
        default=DEFAULT_PREPROCESSING_STEPS,
        help=f"Preprocessing steps passed to {CLI_COMMAND} (default: {' '.join(DEFAULT_PREPROCESSING_STEPS)})",
    )
    parser.add_argument("--rhino-path", help=f"Path to Rhino.exe (default: the {CLI_COMMAND} default)")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args()

    if not args.dry_run and shutil.which(CLI_COMMAND) is None:
        print(f"'{CLI_COMMAND}' not found on PATH. Activate the conda env first: conda activate mesh2nurbs-rhino3d")
        return 1

    failures = []
    for dataset_id in args.dataset_ids:
        failures += convert_dataset(args.dataset_root, dataset_id, args)

    if failures:
        print(f"\nFailed to convert {len(failures)} mesh(es):")
        for path in failures:
            print(f"  - {path}")
        return 1
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
