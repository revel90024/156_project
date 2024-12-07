# Movie Revenue Predictor

A deep learning model that predicts movie box office revenue from movie poster images using PyTorch and ResNet50. The model uses transfer learning and custom architecture to predict the log-transformed revenue values.

## Data Processing Pipeline

### Initial Dataset
- Started with a massive 100k movie poster dataset in torchvision format
- Filtered for movies with revenue > $10M, reducing to ~14k samples
- Removed duplicates based on movie IDs, resulting in 6,137 unique movies
- Final split:
  - Training: 4,999 examples
  - Validation: 556 examples
  - Test: 618 examples

### Data Preprocessing
- Images: Used torchvision's normalization with ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- Revenue:
  - Log10 transformation to handle large value ranges
  - Zero-revenue cases mapped to 0.0

## Model Architecture

### Backbone
- ResNet50 pretrained on ImageNet
- Feature extraction layer outputs 2048-dimensional vector

### Custom Classifier Head
Three-layer architecture:
1. Input Layer: 2048 → 1024
   - ReLU activation
   - Dropout (p=0.2)
2. Hidden Layer: 1024 → 512
   - ReLU activation
   - Dropout (p=0.1)
3. Output Layer: 512 → 1024 → 1
   - Final linear projection to single revenue prediction

### Training Configuration
- Optimizer: AdamW
  - Learning rate: 2e-4
  - Weight decay: 1e-4
- Batch size: 32
- Loss: MSE on log-transformed revenues
- No gradient clipping
- No L1 regularization

### Training Process
- Maximum epochs: 30
- Early stopping:
  - Patience: 7 epochs
  - Monitor: Validation loss
- Learning rate scheduling:
  - ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 2 epochs

## Results Analysis

### Best Model Performance
Achieved at epoch 16:
- Validation loss: 0.2274
- Test metrics:
  - Loss: 0.2399
  - Revenue prediction within 25%: 17.8%
  - Revenue prediction within 50%: 35.6%
  - Binary classification (above/below median):
    - Accuracy: 57.8%
    - Precision: 57.8%
    - Recall: 57.3%
    - F1 Score: 57.6

### Training Progression
- Initial validation loss: 0.6842
- Final validation loss: 0.2274
- Training showed consistent improvement until epoch 16
- Early stopping triggered after epoch 23

### Sample Predictions (from epoch 15)

## Key Findings

1. Model Performance:
   - Best at predicting revenues in the middle range
   - Struggles with extreme outliers (very high/low revenues)
   - Achieves reasonable accuracy for binary classification tasks

2. Architecture Decisions:
   - Three-layer design provided better stability than simpler architectures
   - Varying dropout rates (0.2 → 0.1 → 0.2) helped prevent overfitting
   - Large initial layer (1024) captured complex feature interactions

3. Training Dynamics:
   - Learning plateaued around epoch 16
   - Model showed consistent improvement in early epochs
   - Early stopping prevented overfitting

## Future Improvements

1. Data Augmentation:
   - Implement poster-specific augmentations
   - Balance dataset across revenue ranges

2. Architecture:
   - Experiment with attention mechanisms
   - Try different backbone networks
   - Add genre/metadata features

3. Training:
   - Implement curriculum learning
   - Test different loss functions
   - Explore ensemble methods

**Note**: Movie information in the additional [fun_posters](https://github.com/revel90024/156_project/tree/main/fun_posters) dataset was taken from [IMDB](https://www.imdb.com/). 
