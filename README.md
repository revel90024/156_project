# Movie Revenue Predictor

A deep learning model that predicts movie box office revenue from movie poster images using PyTorch and ResNet50.

## Architecture

### Model (RevenuePredictor)
- Backbone: ResNet50 pretrained on ImageNet
- Custom classifier head:
  - Linear(2048 → 512)
  - ReLU
  - Dropout(0.3)
  - Linear(512 → 1)
- Loss: MSE on log-transformed revenues
- Optimizer: Adam with lr=0.001

### Dataset Processing
- Source: clean_movies_10M/ (filtered from skvarre/movie-posters)
- Filtering criteria:
  - Revenue >= $10M
  - Valid image data
  - Known budget
- Image preprocessing:
  - Resize to 224x224
  - Normalize using ImageNet stats
- Revenue preprocessing:
  - Log transformation (log1p)
  - Handles zero revenues gracefully

## Training

## Implementation Notes

### Data Management
- .gitignore excludes:
  - All datasets (data/, clean_movies_10M/)
  - Model weights (*.pth)
  - Statistics (dataset_stats.json)

### Performance Considerations
- Image preprocessing on CPU
- Model inference on GPU when available
- Batch size tuning per device
- Memory management for large datasets

### Limitations
- Revenue threshold excludes indie films
- Poster-only prediction (no metadata)
- English-language bias in dataset
- Contemporary movie focus

### Future Improvements
1. Model:
   - EfficientNet/ViT alternatives
   - Multi-task learning (revenue + genre)
   - Ensemble predictions

2. Training:
   - Learning rate scheduling
   - Cross-validation
   - Data augmentation

3. Evaluation:
   - Genre-specific performance
   - Time-based analysis
   - Confidence intervals

## Troubleshooting

### Common Issues
1. CUDA/MPS errors:
   - Check torch.cuda.is_available()
   - Verify PyTorch installation
   - Update GPU drivers

2. Memory errors:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use dataset sampling

3. Dataset loading:
   - Verify clean_movies_10M/ path
   - Check disk space
   - Monitor memory usage
