import copy
from pyexpat import features
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2, InterpolationMode
from transformers import CLIPVisionModelWithProjection, CLIPProcessor

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
            batch_features = encoder(images)
            features.append(batch_features.float().cpu().numpy())
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

def make_image_transform(size, mean, std):
    return v2.Compose([
        v2.Resize(
            size,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])


class CLIPImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(
            pixel_values=images,
            interpolate_pos_encoding=True,
        )
        return outputs.image_embeds


def create_feature_encoder(encoder_name, device):
    if encoder_name == "resnet18":
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        encoder.fc = nn.Identity()

        # None means use the existing transform from image_conversion.py.
        transform = None

    elif encoder_name == "dinov2":
        # DINOv2-Small produces 384 features per screenshot.
        encoder = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            trust_repo=True,
        )

        # Both dimensions are divisible by DINOv2's 14-pixel patch size.
        # This also preserves the approximate portrait-screen aspect ratio.
        transform = make_image_transform(
            size=(392, 224),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    elif encoder_name == "clip":

        clip_model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        encoder = CLIPImageEncoder(clip_model)

        # Dimensions are divisible by CLIP's 16-pixel patch size.
        transform = make_image_transform(
            size=(384, 224),
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    encoder = encoder.to(device)
    encoder.eval()

    return encoder, transform

def run_ridge_cross_validation(dataframe, encoder_name="dinov2", batch_size=32, n_splits=5):
    averaged_dataframe = average_image_targets(dataframe)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    encoder, transform = create_feature_encoder(
        encoder_name,
        device,
    )

    if transform is None:
        dataset = UICritImageDataset(averaged_dataframe)
    else:
        dataset = UICritImageDataset(
            averaged_dataframe,
            transform=transform,
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"\nExtracting {encoder_name} features...")

    features, targets = extract_features(
        encoder,
        data_loader,
        device,
    )

    print(
        f"Extracted {features.shape[1]} features "
        f"for {features.shape[0]} screenshots"
    )

    groups = averaged_dataframe["rico_id"].to_numpy()
    alphas = np.logspace(-4, 8, 37)

    splitter = GroupKFold(n_splits=n_splits)

    # Ridge results
    fold_losses = []
    fold_per_target_losses = []

    # Mean-baseline results
    baseline_fold_losses = []
    baseline_per_target_losses = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(
            features,
            targets,
            groups=groups,
        ),
        start=1,
    ):
        # Train Ridge on this fold
        ridge = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=alphas, alpha_per_target=True),
        )

        ridge.fit(
            features[train_indices],
            targets[train_indices],
        )

        # Evaluate Ridge
        predictions = ridge.predict(
            features[validation_indices]
        )

        errors = (
            predictions
            - targets[validation_indices]
        )

        per_target_loss = np.mean(
            errors ** 2,
            axis=0,
        )

        fold_loss = float(
            np.mean(per_target_loss)
        )

        # Calculate mean-predictor baseline using
        # only this fold's training targets
        fold_train_targets = targets[train_indices]
        fold_validation_targets = targets[validation_indices]

        target_means = fold_train_targets.mean(axis=0)

        baseline_predictions = np.tile(
            target_means,
            (len(validation_indices), 1),
        )

        baseline_errors = (
            baseline_predictions
            - fold_validation_targets
        )

        baseline_per_target_loss = np.mean(
            baseline_errors ** 2,
            axis=0,
        )

        baseline_loss = float(
            np.mean(baseline_per_target_loss)
        )

        improvement = (
            (baseline_loss - fold_loss)
            / baseline_loss
            * 100
        )

        # Save this fold's results
        fold_losses.append(fold_loss)
        fold_per_target_losses.append(per_target_loss)

        baseline_fold_losses.append(baseline_loss)
        baseline_per_target_losses.append(
            baseline_per_target_loss
        )

        ridge_target_results = {
            name: round(float(value), 6)
            for name, value in zip(
                TARGET_COLUMNS,
                per_target_loss,
            )
        }

        baseline_target_results = {
            name: round(float(value), 6)
            for name, value in zip(
                TARGET_COLUMNS,
                baseline_per_target_loss,
            )
        }

        print(
            f"Fold {fold}/{n_splits} | "
            f"Baseline MSE: {baseline_loss:.6f} | "
            f"Ridge MSE: {fold_loss:.6f} | "
            f"Improvement: {improvement:.2f}%"
        )

        print(
            f"Baseline per target: "
            f"{baseline_target_results}"
        )

        print(
            f"Ridge per target: "
            f"{ridge_target_results}"
        )

        print(
            f"Selected alphas: {ridge[-1].alpha_}"
        )

    # Calculate averages across all folds
    mean_ridge_loss = float(np.mean(fold_losses))

    mean_baseline_loss = float(np.mean(baseline_fold_losses))

    overall_improvement = ((mean_baseline_loss - mean_ridge_loss) / mean_baseline_loss* 100)

    mean_ridge_per_target = np.mean(fold_per_target_losses, axis=0)

    mean_baseline_per_target = np.mean(baseline_per_target_losses, axis=0)

    ridge_summary = {
        name: round(float(value), 6)
        for name, value in zip(
            TARGET_COLUMNS,
            mean_ridge_per_target,
        )
    }

    baseline_summary = {
        name: round(float(value), 6)
        for name, value in zip(
            TARGET_COLUMNS,
            mean_baseline_per_target,
        )
    }

    print("\nCross-validation summary")

    print(
        f"Mean baseline MSE: "
        f"{mean_baseline_loss:.6f} "
        f"+/- {np.std(baseline_fold_losses, ddof=1):.6f}"
    )

    print(
        f"Mean Ridge MSE: "
        f"{mean_ridge_loss:.6f} "
        f"+/- {np.std(fold_losses, ddof=1):.6f}"
    )

    print(
        f"Overall improvement over baseline: "
        f"{overall_improvement:.2f}%"
    )

    print(
        f"Mean baseline per target: "
        f"{baseline_summary}"
    )

    print(
        f"Mean Ridge per target: "
        f"{ridge_summary}"
    )

    return fold_losses, fold_per_target_losses


def train_model(dataframe, epochs=5, batch_size=32, learning_rate=1e-4):
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
    run_ridge_cross_validation(dataframe, encoder_name="clip", batch_size=8, n_splits=5)

if __name__ == "__main__":
    main()
