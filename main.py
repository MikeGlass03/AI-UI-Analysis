import copy
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

import clean_screenshots
from image_conversion import UICritImageDataset

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "rico_screenshots"

TARGET_COLUMNS = ["learnability", "efficiency", "design_quality_rating"]

# Filters for bad data, i.e. missing images or mismatched comments, and returns a cleaned dataframe with only valid rows.
def load_cleaned_data() -> pd.DataFrame:
    dataframe = pd.read_csv(DATA_DIR / "uicrit_cleaned.csv")
    has_image = dataframe["image_path"].notna()
    has_image &= dataframe["image_path"].map(
        lambda path: isinstance(path, str) and (PROJECT_DIR / path).is_file()
    )
    dataframe[TARGET_COLUMNS] = dataframe[TARGET_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    has_finite_targets = np.isfinite(dataframe[TARGET_COLUMNS]).all(axis=1)
    valid_rows = has_image & has_finite_targets
    dropped_rows = (~valid_rows).sum()
    if dropped_rows:
        print(f"Skipping {dropped_rows} rows with missing images or targets")

    return dataframe.loc[valid_rows].reset_index(drop=True)

# Builds transfer learning model using RestNet18 
def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2), # Switched to 0.2 from 0.5 to reduce overfitting, switch back if it gets worse
        nn.Linear(256, len(TARGET_COLUMNS))
    )
    return model


def train_model(dataframe, epochs=15, batch_size=32, learning_rate=1e-4):
    # Some screenshots have the same rico_id, this makes sure that all screenshots with the same rico_id are in the same split
    first_split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_indices, test_indices = next(first_split.split(dataframe, groups=dataframe["rico_id"]))
    
    second_split = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=43)
    train_indices, validation_indices = next(second_split.split(dataframe.iloc[test_indices], groups=dataframe.iloc[test_indices]["rico_id"]))
    test_indices = dataframe.iloc[test_indices].index

    train_dataframe = dataframe.iloc[train_indices]
    validation_dataframe = dataframe.iloc[validation_indices]
    test_dataframe = dataframe.iloc[test_indices]
    train_loader = DataLoader(UICritImageDataset(train_dataframe), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(UICritImageDataset(validation_dataframe), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(UICritImageDataset(test_dataframe), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": learning_rate * 0.1},
            {"params": model.fc.parameters(), "lr": learning_rate},
        ],
        weight_decay=1e-4,
    )
    loss_function = nn.MSELoss()
    best_validation_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * images.size(0)

        model.eval()
        test_loss = 0.0
        validation_loss = 0.0
        per_target_squared_error = np.zeros(len(TARGET_COLUMNS))
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                predictions = model(images)
                test_loss += loss_function(predictions, targets).item() * images.size(0)
                per_target_squared_error += (
                    (predictions - targets).pow(2).sum(dim=0).cpu().numpy()
                )
            for images, targets in validation_loader:
                images, targets = images.to(device), targets.to(device)
                validation_loss += loss_function(model(images), targets).item() * images.size(0)

        per_target_test_loss = per_target_squared_error / len(test_dataframe)


        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss: {training_loss / len(train_dataframe):.4f} | "
            f"test loss: {test_loss / len(test_dataframe):.4f} | "
            f"validation loss: {validation_loss / len(validation_dataframe):.4f} | "
            f"\nper target test loss: "
            f"{dict(zip(TARGET_COLUMNS, (float(round(value, 4)) for value in per_target_test_loss)))}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_state)
    return model

def main():
    if not IMAGE_DIR.exists():
        print(f"Missing image directory: {IMAGE_DIR}. Would you like to run clean_screenshots.py to download and clean the Rico dataset? (y/n)")
        response = input().strip().lower()
        if response == "y":
            clean_screenshots.main(["--download"])
        else:
            print("Unable to run without Rico screenshots. To manually download, see README instructions.")
            return
    dataframe = load_cleaned_data()
    print(f"Using {len(dataframe)} rows with available screenshots")
    train_model(dataframe)

if __name__ == "__main__":
    main()
