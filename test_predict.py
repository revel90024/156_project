import torch
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import torch.nn as nn
import json
import numpy as np

MODEL_NAME_TO_TEST = "anika_models/anika_dec4.pth"

class RevenuePredictor(nn.Module):
    def __init__(self, hidden_sizes=[1024, 512, 1024], dropout_rates=[0.2, 0.1, 0.2]):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = self.resnet.fc.in_features
        
        layers = []
        in_features = num_features
        
        for hidden_size, dropout_rate in zip(hidden_sizes, dropout_rates):
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size
        
        layers.append(nn.Linear(in_features, 1))
        self.resnet.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.resnet(x)

def load_model():
    model = RevenuePredictor(
        hidden_sizes=[1024, 512, 1024],
        dropout_rates=[0.2, 0.1, 0.2]
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Loading model from: {MODEL_NAME_TO_TEST}")
    model.load_state_dict(torch.load(MODEL_NAME_TO_TEST, weights_only=True))
    model.eval()
    return model, device

def predict_folder(folder_path, movie_data):
    """
    Predict revenues for all images in a folder and compare with actual revenues
    """
    model, device = load_model()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
                    predicted_revenue = float(10 ** log_revenue.squeeze())
                
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

def format_revenue(revenue):
    """Format revenue to handle any size number nicely"""
    if revenue >= 1_000_000_000:  # Billions
        return f"${revenue/1_000_000_000:.1f}B"
    else:  # Millions
        return f"${revenue/1_000_000:.1f}M"

def main():
    # Load movie data
    with open('test_posters/movie-data.json', 'r') as f:
        movie_data = json.load(f)
    
    print(f"\nTesting model: {MODEL_NAME_TO_TEST}")
    print("-" * 80)
    
    predictions = predict_folder('test_posters', movie_data)
    
    # Print results sorted by predicted revenue
    print("\nRanked by PREDICTED Revenue:")
    print("-" * 80)
    print(f"{'Title':<45} {'Predicted':>12} {'Actual':>12} {'Error':>10}")
    print("-" * 80)
    
    predictions_by_predicted = sorted(predictions, key=lambda x: x['predicted_revenue'], reverse=True)
    for pred in predictions_by_predicted:
        print(f"{pred['title']:<45} {format_revenue(pred['predicted_revenue']):>12} "
              f"{format_revenue(pred['actual_revenue']):>12} {pred['error_percent']:>10.1f}%")
    
    # Print results sorted by actual revenue
    print("\nRanked by ACTUAL Revenue:")
    print("-" * 80)
    print(f"{'Title':<45} {'Predicted':>12} {'Actual':>12} {'Error':>10}")
    print("-" * 80)
    
    for r in sorted(predictions, key=lambda x: x['actual_revenue'], reverse=True):
        print(f"{r['title']:<45} {format_revenue(r['predicted_revenue']):>12} "
              f"{format_revenue(r['actual_revenue']):>12} {r['error_percent']:>10.1f}%")

if __name__ == "__main__":
    main()
