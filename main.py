import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Load the dataset
df = pd.read_csv(DATA_DIR / "uicrit_public.csv")

# Regression Target Columns
rating_columns = [
        "aesthetics_rating",
        "learnability",
        "efficiency",
        "usability_rating",
        "design_quality_rating"
]

# Parses list using ast to ensure no lines result in error state
def parse_list(value):
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return None

    return parsed if isinstance(parsed, list) else None

# Cleans the dataset by removing roews in which there is a mismatch between number of comments and actual comments.
def clean_list():
    global df
    source_lists = df['comments_source'].map(parse_list)
    comment_lists = df['comments'].map(parse_list)
    valid_rows = (
        source_lists.notna() 
        & comment_lists.notna() 
        & source_lists.map(len).eq(comment_lists.map(len))
    )

    invalid_row_count = (~valid_rows).sum()
    df = df.loc[valid_rows].copy()

    df = df.rename(columns={'efficency': 'efficiency'})
    df.to_csv(DATA_DIR / "uicrit_cleaned.csv", index=False)

    # Normalize columns
    df["aesthetics_rating"] = (df["aesthetics_rating"] - 1) / 9
    df["learnability"] = (df["learnability"] - 1) / 4
    df["efficiency"] = (df["efficiency"] - 1) / 4
    df["usability_rating"] = (df["usability_rating"] - 1) / 9
    df["design_quality_rating"] = (df["design_quality_rating"] - 1) / 9

    print(f'Removed {invalid_row_count} rows with invalid or mismatched comments.')
    print(f'Cleaned dataset: {len(df)} rows')

def convert_screenshot():
    image_directory = Path(DATA_DIR / "rico_screenshots")

    image_paths = {
        path.stem: str(path)
        for path in image_directory.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    screen_labels["image_path"] = (
        screen_labels["rico_id"]
        .astype(str)
        .map(image_paths)
    )

    print("Missing screenshots:", screen_labels["image_path"].isna().sum())

    screen_labels = screen_labels.dropna(subset=["image_path"])


def train_test_split_data(test_size=0.2, random_state=42):
    X = df[rating_columns]
    y = df[rating_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def main():
    clean_list()
    X_train, X_test, y_train, y_test = train_test_split_data()

    print(f'Training set: {len(X_train)} rows')
    print(f'Testing set: {len(X_test)} rows')

if __name__ == "__main__":
    main()
