import argparse
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CSV_PATH = DATA_DIR / "uicrit_public.csv"
COMBINED_DIR = DATA_DIR / "rico_screenshots"
ASSET_EXTENSIONS = {".jpg", ".json"}


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Remove combined assets whose Rico ID is absent from uicrit_public.csv."
    )
    parser.add_argument(
        "--delete",
        action = "store_true",
        help = "Actually delete unmatched .jpg and .json files; otherwise only preview them.",
    )
    args = parser.parse_args()

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not COMBINED_DIR.is_dir():
        raise FileNotFoundError(f"Combined directory not found: {COMBINED_DIR}")

    valid_ids = load_rico_ids(CSV_PATH)
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
