import argparse
import csv
import shutil
import tarfile
from urllib.request import urlopen
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CSV_PATH = DATA_DIR / "uicrit_public.csv"
COMBINED_DIR = DATA_DIR / "rico_screenshots"
ASSET_EXTENSIONS = {".jpg", ".json"}
RICO_ARCHIVE_URL = (
    "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/"
    "rico_dataset_v0.1/unique_uis.tar.gz"
)
RICO_ARCHIVE_PATH = DATA_DIR / "unique_uis.tar.gz"
RICO_EXTRACT_DIR = DATA_DIR / "rico_screenshots"


def import_rico_dataset(
    rico_ids: set[str],
    url: str = RICO_ARCHIVE_URL,
    archive_path: Path = RICO_ARCHIVE_PATH,
    extract_dir: Path = RICO_EXTRACT_DIR,
) -> None:
    """Download the archive and extract only assets listed in the CSV."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"Downloading RICO archive to {archive_path}...")
        with urlopen(url) as response, archive_path.open("wb") as archive:
            while chunk := response.read(1024 * 1024):
                archive.write(chunk)
        print("Download complete.")
    else:
        print(f"Using existing archive: {archive_path}")

    extracted_count = 0
    print(f"Extracting matching assets to {extract_dir}...")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            member_path = Path(member.name)
            if (not member.isfile() or member_path.suffix.lower() not in ASSET_EXTENSIONS or member_path.stem not in rico_ids):
                continue

            destination = extract_dir / member_path.name
            with archive.extractfile(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted_count += 1

    print(f"Extraction complete. Wrote {extracted_count} matching assets.")


def load_rico_ids(csv_path: Path) -> set[str]:
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "rico_id" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} does not contain a 'rico_id' column")

        return {
            row["rico_id"].strip()
            for row in reader
            if row.get("rico_id") and row["rico_id"].strip()
        }


def find_unmatched_files(valid_ids: set[str], combined_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in combined_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in ASSET_EXTENSIONS
        and path.stem not in valid_ids
    )


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description = "Remove Rico assets whose Rico ID is absent from uicrit_public.csv."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract the RICO UI archive before cleaning.",
    )
    parser.add_argument(
        "--delete",
        action = "store_true",
        help = "Actually delete unmatched .jpg and .json files; otherwise only preview them.",
    )
    args = parser.parse_args(arguments)

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    valid_ids = load_rico_ids(CSV_PATH)

    if args.download:
        import_rico_dataset(valid_ids)

    if not COMBINED_DIR.is_dir():
        raise FileNotFoundError(f"Rico screenshots directory not found: {COMBINED_DIR}")

    unmatched_files = find_unmatched_files(valid_ids, COMBINED_DIR)


    action = "Deleting" if args.delete else "Would delete"
    print(f"Unique valid Rico IDs: {len(valid_ids)}")
    print(f"Unmatched asset files: {len(unmatched_files)}")
    print(f"{action} unmatched files...")

    for path in unmatched_files:
        if args.delete:
            path.unlink()

    if not args.delete:
        for path in unmatched_files[:20]:
            print(path.name)
        if len(unmatched_files) > 20:
            print(f"... and {len(unmatched_files) - 20} more")

        print("Preview only. Run with --delete to remove these files.")


if __name__ == "__main__":
    main()
