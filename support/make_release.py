"""Package TenVoices subset manifests, splits, scripts, and documentation for release.

Rabbi (M2) · Task T26
"""

import argparse
import hashlib
import zipfile
from pathlib import Path
from typing import List, Optional

from support.config import PROJECT_ROOT


def collect_release_files(root: Path) -> List[Path]:
    """Gather all files that belong in the public subset release package.

    Never includes audio or binary model outputs.
    """
    files: List[Path] = []

    # 1. Dataset metadata and manifests
    data_dir = root / "data"
    if (data_dir / "README.md").exists():
        files.append(data_dir / "README.md")
    if (data_dir / "SOURCES.md").exists():
        files.append(data_dir / "SOURCES.md")

    # Subsets (CSV files)
    subsets_dir = data_dir / "subsets"
    if subsets_dir.exists():
        files.extend(sorted(subsets_dir.glob("*.csv")))

    # Prompts
    prompts_dir = data_dir / "prompts"
    if prompts_dir.exists():
        files.extend(sorted(prompts_dir.glob("*.csv")))

    # Splits
    splits_dir = data_dir / "splits"
    if splits_dir.exists():
        files.extend(sorted([p for p in splits_dir.rglob("*.csv") if p.is_file()]))

    # 2. Reproduction scripts
    support_dir = root / "support"
    for script_name in ["select_subsets.py", "qc_report.py"]:
        script_path = support_dir / script_name
        if script_path.exists():
            files.append(script_path)

    return sorted(set(files))


def build_release_package(
    output_path: Optional[Path] = None, root: Optional[Path] = None
) -> Path:
    """Build the zip release package and compute SHA256SUMS.

    Parameters
    ----------
    output_path : Path, optional
        Destination zip path. Defaults to outputs/release/tenvoices-subsets-v1.zip.
    root : Path, optional
        Project root directory. Defaults to PROJECT_ROOT.

    Returns
    -------
    Path
        Path to the generated zip file.
    """
    root_dir = root or PROJECT_ROOT
    if output_path is None:
        target_zip = root_dir / "outputs" / "release" / "tenvoices-subsets-v1.zip"
    else:
        target_zip = output_path

    target_zip.parent.mkdir(parents=True, exist_ok=True)

    files_to_pack = collect_release_files(root_dir)
    if not files_to_pack:
        raise ValueError(f"No release files found in {root_dir}")

    # Compute checksums relative to root
    checksum_lines: List[str] = []
    file_records = []

    for file_path in files_to_pack:
        rel_path = file_path.relative_to(root_dir).as_posix()
        file_bytes = file_path.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        checksum_lines.append(f"{file_hash}  {rel_path}")
        file_records.append((file_path, rel_path))

    checksums_content = "\n".join(checksum_lines) + "\n"

    # Write to zip archive
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add SHA256SUMS file at the root of the archive
        zf.writestr("SHA256SUMS", checksums_content)

        # Add all individual files
        for src_path, arc_name in file_records:
            zf.write(src_path, arcname=arc_name)

    return target_zip


def main():
    parser = argparse.ArgumentParser(description="Package TenVoices subset release.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "release" / "tenvoices-subsets-v1.zip",
        help="Target zip archive path.",
    )
    args = parser.parse_args()

    print(f"Building TenVoices subset release package at: {args.out}...")
    zip_path = build_release_package(output_path=args.out)
    size_kb = zip_path.stat().st_size / 1024.0

    print(f"Successfully generated release package:")
    print(f"  Path: {zip_path}")
    print(f"  Size: {size_kb:.1f} KB")

    with zipfile.ZipFile(zip_path, "r") as zf:
        print(f"  Included files ({len(zf.namelist())}):")
        for name in zf.namelist():
            print(f"    - {name}")


if __name__ == "__main__":
    main()
