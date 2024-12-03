import torch
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import torch.nn as nn
import json
import numpy as np

MODEL_NAME_TO_TEST = "newmodels/best_10M_single.pth"

class RevenuePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.resnet(x)

def predict_folder(folder_path, movie_data):
    """
    Predict revenues for all images in a folder and compare with actual revenues
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = RevenuePredictor()
    model.load_state_dict(torch.load(MODEL_NAME_TO_TEST))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    for image_file in Path(folder_path).glob('*'):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                # Get actual revenue data
                if image_file.name not in movie_data:
                    continue
                    
                movie_info = movie_data[image_file.name]
                actual_revenue = movie_info['actual_revenue']
                title = movie_info['title']
                
                # Predict revenue
                image = Image.open(image_file).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    log_revenue = model(image_tensor)
                    predicted_revenue = torch.exp(log_revenue).item() - 1
                
                # Calculate error
                error_percent = abs(predicted_revenue - actual_revenue) / actual_revenue * 100
                
                results.append({
                    'title': title,
                    'predicted_revenue': predicted_revenue,
                    'actual_revenue': actual_revenue,
                    'error_percent': error_percent
                })
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {str(e)}")
    
    return results

def main():
    # Load movie data
    with open('test_posters/movie-data.json', 'r') as f:
        movie_data = json.load(f)
    
    print(f"\nTesting model: {MODEL_NAME_TO_TEST}")
    print("-" * 80)
    
    # Make predictions
    predictions = predict_folder('test_posters', movie_data)
    
    # Calculate overall metrics
    errors = [p['error_percent'] for p in predictions]
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    within_25 = sum(1 for e in errors if e <= 25) / len(errors) * 100
    within_50 = sum(1 for e in errors if e <= 50) / len(errors) * 100
    
    # Sort by actual revenue
    predictions.sort(key=lambda x: x['actual_revenue'], reverse=True)
    
    # Print results
    print("\nPredictions (sorted by actual revenue):")
    print("-" * 80)
    print(f"{'Title':<35} {'Predicted':<15} {'Actual':<15} {'Error':<10}")
    print("-" * 80)
    
    for pred in predictions:
        print(f"{pred['title']:<35} ${pred['predicted_revenue']:>14,.0f} ${pred['actual_revenue']:>14,.0f} {pred['error_percent']:>7.1f}%")
    
    print("\nOverall Metrics:")
    print("-" * 80)
    print(f"Mean Error:     {mean_error:.1f}%")
    print(f"Median Error:   {median_error:.1f}%")
    print(f"Within 25%:     {within_25:.1f}%")
    print(f"Within 50%:     {within_50:.1f}%")
    print("-" * 80)

if __name__ == "__main__":
    main()
