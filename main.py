# Standard imports - could also consider using fastai for higher-level abstractions
# or tensorflow/keras for a different framework altogether
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image
import numpy as np
from pathlib import Path
import os
import shutil

# Setup paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Remove the existing incomplete directory if it exists
if (DATA_DIR / "skvarre___movie-posters").exists():
    shutil.rmtree(DATA_DIR / "skvarre___movie-posters")

print("Downloading dataset to:", DATA_DIR)
# Download dataset directly to project directory
dataset = load_dataset(
    "skvarre/movie-posters",
    cache_dir=DATA_DIR
)

print("\nDataset structure:", dataset)
print("Number of examples:", len(dataset['train']))
print("\nSample data point:", dataset['train'][0])

print("\nDebug Info:")
print("Current directory:", PROJECT_ROOT)
print("Data directory:", DATA_DIR)
print("Does data directory exist?:", DATA_DIR.exists())
print("Contents of data directory:", list(DATA_DIR.glob("*")))

# Debug directory contents
dataset_path = DATA_DIR / "skvarre___movie-posters"
print("\nChecking directory contents:")
print(f"Directory exists: {dataset_path.exists()}")
print("Contents:")
for root, dirs, files in os.walk(dataset_path):
    print(f"\nIn {root}:")
    print("Directories:", dirs)
    print("Files:", files)

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
        
        # Debug print for first few items
        if idx < 5:
            print(f"Raw revenue for movie '{item['title']}': ${item['revenue']:,}")
        
        image = item['image']
        revenue = float(item['revenue'])
        
        # Skip log transform if revenue is 0
        if revenue > 0:
            revenue = np.log1p(revenue)
        else:
            revenue = 0.0  # or some small value like 1.0
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(revenue, dtype=torch.float32)

class RevenuePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Update to use newer weights parameter
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace the last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.resnet(x)

def train_model(model, train_loader, val_loader, epochs=10):
    # Check for MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU!")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Track metrics
    best_loss = float('inf')
    epoch_losses = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        epoch_loss = 0
        batch_count = len(train_loader)
        
        for batch_idx, (images, revenues) in enumerate(train_loader):
            images, revenues = images.to(device), revenues.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, revenues.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                with torch.no_grad():
                    print(f"\nBatch {batch_idx}")
                    print(f"Loss: {loss.item():.4f}")
                    print("\nSample predictions vs actuals:")
                    for i in range(3):
                        raw_pred = torch.exp(outputs[i]).item() - 1
                        raw_actual = torch.exp(revenues[i]).item() - 1
                        log_pred = outputs[i].item()
                        log_actual = revenues[i].item()
                        
                        print(f"Movie {i+1}:")
                        print(f"  Predicted (raw): ${raw_pred:,.0f}")
                        print(f"  Actual (raw):    ${raw_actual:,.0f}")
                        print(f"  Predicted (log): {log_pred:.2f}")
                        print(f"  Actual (log):    {log_actual:.2f}")
                        print(f"  Log-space error: {abs(log_pred - log_actual):.2f}")
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / batch_count
        epoch_losses.append(avg_epoch_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        if len(epoch_losses) > 1:
            improvement = (epoch_losses[-2] - avg_epoch_loss) / epoch_losses[-2] * 100
            print(f"Improvement from last epoch: {improvement:.1f}%")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'best_model_1M.pth')
            print("✓ Saved new best model")

def prepare_dataset(dataset, min_revenue=100_000, train_percentage=1.0):
    """
    1. Convert to list once
    2. Sample from list (fast)
    3. Filter revenue
    """
    print("\nPreparing dataset...")
    
    # Convert to list once
    print("Converting dataset to list...")
    dataset_list = list(dataset)
    total_size = len(dataset_list)
    
    # Sample
    sample_size = int(total_size * train_percentage)
    print(f"Sampling {train_percentage*100:.0f}% of data ({sample_size:,d} movies)...")
    import random
    random.seed(42)
    sampled_data = random.sample(dataset_list, sample_size)
    
    # Filter
    print(f"Filtering movies below ${min_revenue:,} revenue...")
    filtered_data = [
        movie for movie in sampled_data 
        if movie['revenue'] >= min_revenue
    ]
    
    print(f"Final dataset size: {len(filtered_data):,d} movies")
    
    print("Converting back to Dataset format...")
    from datasets import Dataset
    return Dataset.from_list(filtered_data)

def main():
    print("Loading dataset...")
    dataset = load_dataset("skvarre/movie-posters", cache_dir=DATA_DIR)
    
    clean_subset = prepare_dataset(
        dataset["train"],
        min_revenue=1_000_000,
        train_percentage=1.0  # Try with 10% first
    )
    
    print("Splitting into train/val/test...")
    splits = clean_subset.train_test_split(test_size=0.2, seed=42)
    print("Creating validation split...")
    train_val = splits["train"].train_test_split(test_size=0.1, seed=42)
    
    # Create datasets
    train_dataset = MoviePosterDataset(train_val, split="train")
    val_dataset = MoviePosterDataset(train_val, split="test")
    test_dataset = MoviePosterDataset({"train": splits["test"]}, split="train")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Print split sizes
    print(f"\nDataset splits:")
    print(f"Training:   {len(train_dataset):,d} examples")
    print(f"Validation: {len(val_dataset):,d} examples")
    print(f"Test:      {len(test_dataset):,d} examples")
    
    model = RevenuePredictor()
    train_model(model, train_loader, val_loader)
    
    # Save model for later testing
    torch.save(model.state_dict(), 'revenue_predictor.pth')

if __name__ == "__main__":
    main()