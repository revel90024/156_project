import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from datasets import load_from_disk
import numpy as np
import os
import json
from PIL import Image
from full_predict import evaluate_sample

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
            
        return image, torch.tensor(revenue, dtype=torch.float32)

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

def train_with_params(train_loader, val_loader, params):
    model = RevenuePredictor()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on: {device}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    early_stopping_count = 0
    
    for epoch in range(params['epochs']):
        # Training phase
        model.train()
        train_losses = []
        
        for images, revenues in train_loader:
            images, revenues = images.to(device), revenues.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = nn.MSELoss()(outputs, revenues.unsqueeze(1))
            
            if params['l1_lambda'] > 0:
                l1_loss = sum(p.abs().sum() for p in model.parameters())
                loss += params['l1_lambda'] * l1_loss
                
            loss.backward()
            
            if params['clip_grad']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip_grad'])
                
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, revenues in val_loader:
                images, revenues = images.to(device), revenues.to(device)
                outputs = model(images)
                val_loss = nn.MSELoss()(outputs, revenues.unsqueeze(1))
                val_losses.append(val_loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        print(f"\nEpoch {epoch+1}/{params['epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/clean_movies_1M_modern_best.pth')
            early_stopping_count = 0
        else:
            early_stopping_count += 1
            if early_stopping_count >= params['patience']:
                print("Early stopping triggered")
                break
    
    return best_val_loss

def grid_search(train_loader, val_loader):
    param_grid = {
        'learning_rate': [1e-3, 5e-4, 1e-4],
        'weight_decay': [0.01, 0.001],
        'epochs': [20],
        'patience': [5],
        'l1_lambda': [0, 0.001],
        'clip_grad': [None, 1.0],
        'batch_size': [16, 32]
    }
    
    results = []
    
    from itertools import product
    keys = param_grid.keys()
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        print("\nTrying parameters:", params)
        
        val_loss = train_with_params(train_loader, val_loader, params)
        results.append((val_loss, params))
        
        # Save results after each trial
        results.sort(key=lambda x: x[0])
        with open('models/clean_movies_1M_modern_tuning.json', 'w') as f:
            json.dump({
                'best_val_loss': results[0][0],
                'best_params': results[0][1],
                'all_results': [
                    {'val_loss': loss, 'params': params}
                    for loss, params in results
                ]
            }, f, indent=4)

def main():
    os.makedirs('models', exist_ok=True)
    
    print("Loading clean dataset...")
    dataset = load_from_disk("clean_movies_1M_modern")
    
    print("Splitting into train/val/test...")
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_val = splits["train"].train_test_split(test_size=0.1, seed=42)
    
    # Create datasets
    train_dataset = MoviePosterDataset(train_val, split="train")
    val_dataset = MoviePosterDataset(train_val, split="test")
    
    # Create initial loaders with default batch size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print(f"\nDataset splits:")
    print(f"Training:   {len(train_dataset):,d} examples")
    print(f"Validation: {len(val_dataset):,d} examples")
    
    print("\nStarting hyperparameter search...")
    grid_search(train_loader, val_loader)
    
    print("\nEvaluating best model on test set...")
    evaluate_sample(splits["test"])  # Pass the test split

if __name__ == "__main__":
    main()