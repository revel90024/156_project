import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from datasets import load_from_disk
import numpy as np
import random
from PIL import Image

class MoviePosterDataset(Dataset):
    def __init__(self, hf_dataset, split="train", transform=None):
        self.dataset = hf_dataset[split]
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        revenue = float(item['revenue'])
        
        if revenue > 0:
            revenue = np.log1p(revenue)
        else:
            revenue = 0.0
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(revenue, dtype=torch.float32), item['title']

class RevenuePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.resnet(x)

def evaluate_sample(dataset=None):
    print("Loading dataset...")
    if dataset is None:
        dataset = load_from_disk("clean_movies_1M_modern")
    
    # Take random sample of 300 movies
    all_indices = list(range(len(dataset)))
    sample_indices = random.sample(all_indices, 300)
    sample_dataset = dataset.select(sample_indices)
    
    # Create dataloader
    test_dataset = MoviePosterDataset({"train": sample_dataset}, split="train")
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print("Loading best model from hyperparameter tuning...")
    model = RevenuePredictor()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.load_state_dict(torch.load('newarchitecture/best_10M_single_v1.pth'))
    model = model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    titles = []
    
    print("\nEvaluating sample of 100 movies...")
    with torch.no_grad():
        for images, revenues, movie_titles in test_loader:
            images = images.to(device)
            revenues = revenues.to(device)
            outputs = model(images)
            
            # Convert back to raw revenues
            pred_revenue = torch.exp(outputs) - 1
            actual_revenue = torch.exp(revenues.unsqueeze(1)) - 1
            
            predictions.extend(pred_revenue.cpu().numpy())
            actuals.extend(actual_revenue.cpu().numpy())
            titles.extend(movie_titles)
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Percentage error
    errors = np.abs(predictions - actuals) / actuals * 100
    
    # Revenue brackets for accuracy
    brackets = {
        "Low": (0, 10_000_000),
        "Medium": (10_000_000, 100_000_000),
        "High": (100_000_000, float('inf'))
    }
    
    bracket_accuracy = {name: [] for name in brackets}
    
    print("\nDetailed Results:")
    print("-" * 80)
    for i in range(min(100, len(predictions))):
        pred = predictions[i][0]
        actual = actuals[i][0]
        error = errors[i][0]
        title = titles[i]
        
        # Find revenue bracket
        for bracket_name, (low, high) in brackets.items():
            if low <= actual < high:
                # Consider prediction correct if within 50% of actual
                bracket_accuracy[bracket_name].append(error <= 50)
                break
        
        print(f"\nMovie: {title}")
        print(f"Predicted: ${pred:,.2f}")
        print(f"Actual:    ${actual:,.2f}")
        print(f"Error:     {error:.1f}%")
    
    print("\nOverall Metrics:")
    print("-" * 80)
    print(f"Mean Absolute Error: ${np.mean(np.abs(predictions - actuals)):,.2f}")
    print(f"Median Error: {np.median(errors):.1f}%")
    print(f"Within 25% of actual: {np.mean(errors <= 25)*100:.1f}%")
    print(f"Within 50% of actual: {np.mean(errors <= 50)*100:.1f}%")
    
    print("\nAccuracy by Revenue Bracket:")
    print("-" * 80)
    for bracket, correct in bracket_accuracy.items():
        if correct:  # Only show if bracket has movies
            accuracy = np.mean(correct) * 100
            count = len(correct)
            print(f"{bracket} Budget Films ({count} movies): {accuracy:.1f}% within 50% of actual")

if __name__ == "__main__":
    evaluate_sample()