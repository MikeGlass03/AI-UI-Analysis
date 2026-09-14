# AI UI Analysis

This project evaluates whether pretrained computer-vision representations
can predict human-rated UI quality from screenshots. The project compares
ResNet18, CLIP ViT-B/16, and Dinov2 image features using Ridge regression
across learnability, efficiency, and overall design targets.

The central idea is that human-based ratings of various different metrics
could be used to train and generalize UI screenshots to provide feedback
that matched or was similar to the dataset. This project's focus was
inspired primarily by this research paper:

UICrit: Enhancing Automated Design Evaluation with a UICritique Dataset
Peitong Duan, Chin-yi Chen, Gang Li, Bjoern Hartmann, Yang Li

Their paper focused on human and LLM generated comments, whereas this
project focuses primarily on generating accurate numbers for rating on
UICrit metrics.

Dataset ratings are derived from the UICrit dataset and corresponding 
screenshots from RICO. Both datasets remain the work of their respective 
authors and are used according to their original terms.

## Results

The best model used DINOv2-Small CLS and mean patch features with
StandardScaler and RidgeCV.

| Model | Mean CV MSE | Improvement from baseline |
|---|---:|---:|
| Mean baseline | 0.007385 | N/A |
| ResNet18 | 0.007190 | 2.65% |
| CLIP ViT-B/16 | 0.007207 | 2.42% |
| DINOv2 CLS | 0.007145 | 3.25% |
| DINOv2 CLS + patch mean | 0.007124 | 3.53% |

Improvement is calculated relative to a fold-specific baseline that
predicts the training fold's mean rating. All targets were normalized
to the range 0–1.

## Problem

Visual appearance alone may provide useful information about the
quality of a user interface.This project investigates whether
pretrained computer-vision models can extract enough information from
a UI screenshot to predict:

- Learnability
- Efficiency
- Overall design quality

Success in this project would allow for screenshots to be submitted
to this model to get a review of user interfaces. This requires the
CNN models being trained on to generalize to this specific set of
UI screenshots and the associated human ratings given to them.

## Dataset

This project combines design ratings from UICrit with corresponding
screenshots from the RICO mobile-interface dataset.

After matching screenshots, removing invalid rows, and averaging
multiple annotations for each interface, the model uses:

- 2,897 valid rating rows
- 1,000 unique UI screenshots
- Three prediction targets

The datasets were created by their respective authors and are not
original work from this repository.

### Expected data structure

```text
AI-UI-Analysis/
├── clean_screenshots.py
├── image_conversion.py
├── main.py
├── data/
│   ├── uicrit_cleaned.csv
│   ├── uicrit_public.csv
│   └── rico_screenshots/
│       └── [screenshot files]  # Excluded from GitHub
├── README.md
└── requirements.txt
'''

## Method

1. Match UICrit records to RICO screenshots using `rico_id`.
2. Remove records with missing screenshots or targets.
3. Average multiple ratings belonging to the same screenshot.
4. Resize and normalize each screenshot for its selected encoder.
5. Extract frozen image features using ResNet18, CLIP, or DINOv2.
6. Standardize the extracted features.
7. Fit multi-output Ridge regression
8. Select regularization strengths using RidgeCV.

The final DINOv2 representation concatenates:

- A 384-dimensional CLS feature representing the complete screenshot.
- A 384-dimensional mean patch feature representing local UI content.

This produces 768 features per screenshot.

## Evaluation

Models are evaluated using five-fold cross-validation grouped by
`rico_id`.

For each fold:

- The model is trained on four folds.
- The remaining fold is used only for evaluation.
- The baseline uses only the mean targets from the training portion.
- Mean squared error is calculated overall and separately for each
  target.

The same folds are used for every encoder comparison.

## Installation

```powershell
git clone https://github.com/MikeGlass03/AI-UI-Analysis.git
cd AI-UI-Analysis

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python main.py

To change the encoder being used, edit line 484:

run_ridge_cross_validation(dataframe, encoder_name="dinov2", batch_size=32, n_splits=5)

by replacing encoder_name="" with one of the following:
clip
dinov2
resnet18

To install the RICO screenshots (which are not included in this
Repository due to size and licensing), please either use the script
provided on initial run. If you would like to do it manually, install
the screenshots here: https://www.interactionmining.org/archive/rico,
then to clean/delete unneccesary screenshots, run:

python image_conversion.py --delete.
