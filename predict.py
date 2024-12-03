import torch
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import torch.nn as nn

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

def predict_folder(folder_path):
    """
    Predict revenues for all images in a folder
    """
    # Setup model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = RevenuePredictor()
    model.load_state_dict(torch.load(MODEL_NAME_TO_TEST))  # Updated path
    model.to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Process each image
    results = []
    for image_file in Path(folder_path).glob('*'):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                # Load and transform image
                image = Image.open(image_file).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    log_revenue = model(image_tensor)
                    predicted_revenue = torch.exp(log_revenue).item() - 1
                
                results.append({
                    'filename': image_file.name,
                    'predicted_revenue': predicted_revenue
                })
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {str(e)}")
    
    return results

def main():
    # Paths
    image_folder = 'test_posters'  # Folder containing test images
    
    print(f"\nPredicting revenues for images in: {image_folder}")
    print("-" * 50)
    
    # Make predictions
    predictions = predict_folder(image_folder)
    
    # Sort by predicted revenue
    predictions.sort(key=lambda x: x['predicted_revenue'], reverse=True)
    
    # Print results
    print("\nPredictions:")
    print("-" * 50)
    for pred in predictions:
        print(f"{pred['filename']:<30} ${pred['predicted_revenue']:,.2f}")

if __name__ == "__main__":
    main()
