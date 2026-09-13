import copy
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
    dataframe[TARGET_COLUMNS] = dataframe[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    has_finite_targets = np.isfinite(dataframe[TARGET_COLUMNS]).all(axis=1)
    valid_rows = has_image & has_finite_targets
    dropped_rows = (~valid_rows).sum()
    if dropped_rows:
        print(f"Skipping {dropped_rows} rows with missing images or targets")

    return dataframe.loc[valid_rows].reset_index(drop=True)

# I think this will optimize results better, because there are 1-3 ratings per screenshot
def average_image_targets(dataframe):
    return dataframe.groupby(["rico_id", "image_path"], as_index=False)[TARGET_COLUMNS].mean()

# Builds transfer learning model using RestNet18 
def build_model(target_means) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for parameter in model.parameters():
        parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, len(TARGET_COLUMNS))

    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.copy_(
            torch.tensor(target_means, dtype=torch.float32)
        )

    return model


def evaluate_model(model, data_loader, device):
    model.eval()
    loss_function = nn.MSELoss()
    total_loss = 0.0
    per_target_squared_error = np.zeros(len(TARGET_COLUMNS))
    total_samples = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            total_loss += loss_function(predictions, targets).item() * images.size(0)
            per_target_squared_error += (
                (predictions - targets).pow(2).sum(dim=0).cpu().numpy()
            )
            total_samples += images.size(0)

    return (
        total_loss / total_samples,
        per_target_squared_error / total_samples,
    )

def extract_features(encoder, data_loader, device):
    features = []
    targets = []

    encoder.eval()
    with torch.no_grad():
        for images, batch_targets in data_loader:
            images = images.to(device)
            features.append(encoder(images).cpu().numpy())
            targets.append(batch_targets.numpy())

    return np.concatenate(features), np.concatenate(targets)


def run_ridge_regression(dataframe, batch_size=32):
    first_split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_indices, test_indices = next(
        first_split.split(dataframe, groups=dataframe["rico_id"])
    )
    second_split = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=43)
    train_relative_indices, validation_relative_indices = next(
        second_split.split(
            dataframe.iloc[train_val_indices],
            groups=dataframe.iloc[train_val_indices]["rico_id"],
        )
    )
    train_indices = train_val_indices[train_relative_indices]
    validation_indices = train_val_indices[validation_relative_indices]

    train_dataframe = average_image_targets(dataframe.iloc[train_indices])
    validation_dataframe = average_image_targets(dataframe.iloc[validation_indices])
    test_dataframe = average_image_targets(dataframe.iloc[test_indices])

    train_loader = DataLoader(UICritImageDataset(train_dataframe), batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(UICritImageDataset(validation_dataframe), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(UICritImageDataset(test_dataframe), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder.fc = nn.Identity()
    encoder = encoder.to(device)
    encoder.eval()

    train_features, train_targets = extract_features(encoder, train_loader, device)
    validation_features, validation_targets = extract_features(encoder, validation_loader, device)
    test_features, test_targets = extract_features(encoder, test_loader, device)

    alphas = np.logspace(-4, 8, 37)
    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, alpha_per_target=True),
    )
    ridge.fit(train_features, train_targets)

    validation_predictions = ridge.predict(validation_features)
    test_predictions = ridge.predict(test_features)
    validation_loss = np.mean((validation_predictions - validation_targets) ** 2)
    test_loss = np.mean((test_predictions - test_targets) ** 2)
    per_target_test_loss = np.mean((test_predictions - test_targets) ** 2, axis=0)

    print(f"Ridge validation loss: {validation_loss:.6f}")
    print(f"Ridge test loss: {test_loss:.6f}")
    print("Ridge per target test loss: "f"{dict(zip(TARGET_COLUMNS, per_target_test_loss.round(6).tolist()))}")
    print(f"Ridge selected alphas: {ridge[-1].alpha_}")

    return ridge


def run_ridge_cross_validation(dataframe, batch_size=32, n_splits=5):
    averaged_dataframe = average_image_targets(dataframe)
    data_loader = DataLoader(
        UICritImageDataset(averaged_dataframe),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder.fc = nn.Identity()
    encoder = encoder.to(device)
    encoder.eval()

    features, targets = extract_features(encoder, data_loader, device)
    groups = averaged_dataframe["rico_id"].to_numpy()
    alphas = np.logspace(-4, 8, 37)
    splitter = GroupKFold(n_splits=n_splits)
    fold_losses = []
    fold_per_target_losses = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, targets, groups=groups), start=1
    ):
        ridge = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=alphas, alpha_per_target=True),
        )
        ridge.fit(features[train_indices], targets[train_indices])
        predictions = ridge.predict(features[validation_indices])
        errors = predictions - targets[validation_indices]
        per_target_loss = np.mean(errors ** 2, axis=0)
        fold_loss = float(np.mean(per_target_loss))
        fold_losses.append(fold_loss)
        fold_per_target_losses.append(per_target_loss)

        print(
            f"Ridge fold {fold}/{n_splits} | "
            f"MSE: {fold_loss:.6f} | "
            f"per target: "
            f"{dict(zip(TARGET_COLUMNS, per_target_loss.round(6).tolist()))} | "
            f"alphas: {ridge[-1].alpha_}"
        )

    mean_per_target_loss = np.mean(fold_per_target_losses, axis=0)
    print(
        f"Ridge {n_splits}-fold mean MSE: "
        f"{np.mean(fold_losses):.6f} +/- {np.std(fold_losses):.6f}"
    )
    print(
        "Ridge mean per target MSE: "
        f"{dict(zip(TARGET_COLUMNS, mean_per_target_loss.round(6).tolist()))}"
    )

    return fold_losses, fold_per_target_losses


def train_model(dataframe, epochs=15, batch_size=32, learning_rate=1e-4):
    # Some screenshots have the same rico_id, this makes sure that all screenshots with the same rico_id are in the same split
    first_split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_indices, test_indices = next(first_split.split(dataframe, groups=dataframe["rico_id"]))
    
    second_split = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=43)
    train_relative_indices, validation_relative_indices = next(second_split.split(dataframe.iloc[train_val_indices], groups=dataframe.iloc[train_val_indices]["rico_id"]))
    train_indices = train_val_indices[train_relative_indices]
    validation_indices = train_val_indices[validation_relative_indices]

    train_dataframe = average_image_targets(dataframe.iloc[train_indices])
    validation_dataframe = average_image_targets(dataframe.iloc[validation_indices])
    test_dataframe = average_image_targets(dataframe.iloc[test_indices])
    
    train_loader = DataLoader(UICritImageDataset(train_dataframe), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(UICritImageDataset(validation_dataframe), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(UICritImageDataset(test_dataframe), batch_size=batch_size)
    
    target_means = train_dataframe[TARGET_COLUMNS].mean().to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(target_means).to(device)
    
    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=learning_rate,
        weight_decay=0.05,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    
    loss_function = nn.MSELoss()
    best_validation_loss, _ = evaluate_model(model, validation_loader, device)
    best_model_state = copy.deepcopy(model.state_dict())
    print(f"Initial validation loss: {best_validation_loss:.6f}")
    
    for epoch in range(epochs):
        model.eval()
        model.fc.train()
        training_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()

            predictions = model(images)
            loss = loss_function(predictions, targets)

            loss.backward()
            optimizer.step()

            training_loss += loss.item() * images.size(0)

        average_training_loss = training_loss / len(train_dataframe)

        # Validation is used to select the best epoch.
        average_validation_loss, _ = evaluate_model(
            model,
            validation_loader,
            device,
        )

        scheduler.step(average_validation_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {average_training_loss:.6f} | "
            f"Validation loss: {average_validation_loss:.6f}"
        )

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            best_model_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_state)
    final_test_loss, final_per_target_loss = evaluate_model(model, test_loader, device)
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Final test loss: {final_test_loss:.6f}")

    print("Final per-target test loss:",dict(zip(TARGET_COLUMNS,final_per_target_loss.round(6).tolist())))
    
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
    run_ridge_cross_validation(dataframe)

if __name__ == "__main__":
    main()
