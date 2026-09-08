import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from sklearn.model_selection import train_test_split
from pathlib import Path

from image_conversion import UICritImageDataset

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "rico_screenshots"

TARGET_COLUMNS = ["learnability", "efficiency", "design_quality_rating"]


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


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(TARGET_COLUMNS))
    return model


def train_model(dataframe, epochs=5, batch_size=32, learning_rate=1e-3):
    train_dataframe, test_dataframe = train_test_split(
        dataframe, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(
        UICritImageDataset(train_dataframe), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(UICritImageDataset(test_dataframe), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

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
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                test_loss += loss_function(model(images), targets).item() * images.size(0)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss: {training_loss / len(train_dataframe):.4f} | "
            f"test loss: {test_loss / len(test_dataframe):.4f}"
        )

    return model

def main():
    dataframe = load_cleaned_data()
    print(f"Using {len(dataframe)} rows with available screenshots")
    train_model(dataframe)

if __name__ == "__main__":
    main()
