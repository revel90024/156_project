# Movie Revenue Predictor

A deep learning model that predicts movie box office revenue from movie poster images using PyTorch and ResNet50.

Note that the model .pth files and the datasets are too large so haven't pushed them to the repo. 10M is dataset of movies with revenue over 10M, modern indicates movies newer than 2000.
Its kind of a mess but I think I might be onto something with this new architecture. If we can get the validation loss down to below 1.0 I think we're good.

## Project Structure

### Core Files
- `image_only.py`: Main training script for poster-only model
  - Implements single training runs
  - Handles early stopping
  - Saves model checkpoints and metrics
  - Current best architecture: [1024, 512] with 0.2 dropout

- `all_features.py`: Enhanced model with additional features
  - Includes metadata (budget, genre, etc.)
  - More sophisticated training pipeline
  - Better metrics tracking
  - Currently experimental

- `test_predict.py`: Evaluation script
  - Tests models on new posters
  - Provides detailed error analysis
  - Ranks predictions by different criteria

### Supporting Files
- `clean_dataset.py`: Data preprocessing
  - Filters movies by revenue threshold
  - Handles image validation
  - Generates dataset statistics

- `main.py`: Legacy training script
  - Original implementation
  - Simpler architecture [512] with 0.3 dropout
  - Kept for comparison

### Model Versions
- `newmodels/best_10M_single.pth`: Original model
  - Single hidden layer (512)
  - 0.3 dropout
  - ~1.16 validation loss

- `newarchitecture/best_10M_single_v1.pth`: Current best
  - Two hidden layers (1024, 512)
  - 0.2 dropout each
  - Better performance on high-budget films

## Model Architecture

### Current Best (v3)
- Backbone: ResNet50 (pretrained)
- Classifier head:
  - Linear(2048 → 1024)
  - ReLU + Dropout(0.2)
  - Linear(1024 → 512)
  - ReLU + Dropout(0.2)
  - Linear(512 → 1)
- Training:
  - Loss: MSE on log-transformed revenues
  - Optimizer: AdamW (lr=2e-4)
  - Early stopping (patience=7)
  - LR scheduling (ReduceLROnPlateau)

## Usage

1. Training a new model:
