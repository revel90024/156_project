import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from datasets import load_from_disk
import numpy as np
import os
import json
from PIL import Image
from datetime import datetime

MODEL_NAME = "best_10M_single"
DATASET_NAME = "clean_movies_10M"

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
    
    def print_predictions(model, loader, num_examples=5):
        model.eval()
        with torch.no_grad():
            images, revenues = next(iter(loader))
            images, revenues = images.to(device), revenues.to(device)
            outputs = model(images)
            
            # Convert back from log space
            pred_revenue = torch.exp(outputs.squeeze()) - 1
            actual_revenue = torch.exp(revenues) - 1
            
            print("\nSample Predictions:")
            print("-" * 50)
            for i in range(num_examples):
                error_percent = abs(pred_revenue[i].item() - actual_revenue[i].item()) / actual_revenue[i].item() * 100
                print(f"Example {i+1}:")
                print(f"Predicted: ${pred_revenue[i].item():,.0f}")
                print(f"Actual:    ${actual_revenue[i].item():,.0f}")
                print(f"Error:     {error_percent:.1f}%")
                print("-" * 50)
    
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
        all_preds = []
        all_actuals = []
        
        with torch.no_grad():
            for images, revenues in val_loader:
                images, revenues = images.to(device), revenues.to(device)
                outputs = model(images)
                val_loss = nn.MSELoss()(outputs, revenues.unsqueeze(1))
                val_losses.append(val_loss.item())
                
                # Store predictions and actuals for metrics
                pred_revenue = torch.exp(outputs.squeeze()) - 1
                actual_revenue = torch.exp(revenues) - 1
                all_preds.extend(pred_revenue.cpu().numpy())
                all_actuals.extend(actual_revenue.cpu().numpy())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Calculate regression metrics
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)
        within_25 = np.mean(np.abs(all_preds - all_actuals) / all_actuals <= 0.25) * 100
        within_50 = np.mean(np.abs(all_preds - all_actuals) / all_actuals <= 0.50) * 100
        
        # Calculate classification metrics (above/below median)
        median_revenue = np.median(all_actuals)
        pred_high = all_preds > median_revenue
        actual_high = all_actuals > median_revenue
        
        accuracy = np.mean(pred_high == actual_high) * 100
        
        # Handle division by zero cases
        if np.sum(pred_high) > 0:
            precision = np.sum(pred_high & actual_high) / np.sum(pred_high) * 100
        else:
            precision = 0
            
        if np.sum(actual_high) > 0:
            recall = np.sum(pred_high & actual_high) / np.sum(actual_high) * 100
        else:
            recall = 0
            
        # Calculate F1 Score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        print(f"\nEpoch {epoch+1}/{params['epochs']}")
        print("-" * 50)
        print(f"Train Loss:    {avg_train_loss:.4f}")
        print(f"Val Loss:      {avg_val_loss:.4f}")
        print(f"Within 25%:    {within_25:.1f}%")
        print(f"Within 50%:    {within_50:.1f}%")
        print(f"Accuracy:      {accuracy:.1f}%")
        print(f"Precision:     {precision:.1f}%")
        print(f"Recall:        {recall:.1f}%")
        print(f"F1 Score:      {f1_score:.1f}")
        print("-" * 50)
        
        # Print sample predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            print_predictions(model, val_loader)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'newmodels/{MODEL_NAME}.pth')
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
        with open(f'newmodels/{MODEL_NAME}.json', 'w') as f:
            json.dump({
                'best_val_loss': results[0][0],
                'best_params': results[0][1],
                'all_results': [
                    {'val_loss': loss, 'params': params}
                    for loss, params in results
                ]
            }, f, indent=4)

def evaluate_on_test(model, test_loader, device):
    model.eval()
    test_losses = []
    all_preds = []
    all_actuals = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, revenues in test_loader:
            images, revenues = images.to(device), revenues.to(device)
            outputs = model(images)
            test_loss = nn.MSELoss()(outputs, revenues.unsqueeze(1))
            test_losses.append(test_loss.item())
            
            # Store predictions and actuals for metrics
            pred_revenue = torch.exp(outputs.squeeze()) - 1
            actual_revenue = torch.exp(revenues) - 1
            all_preds.extend(pred_revenue.cpu().numpy())
            all_actuals.extend(actual_revenue.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    
    within_25 = np.mean(np.abs(all_preds - all_actuals) / all_actuals <= 0.25) * 100
    within_50 = np.mean(np.abs(all_preds - all_actuals) / all_actuals <= 0.50) * 100
    
    median_revenue = np.median(all_actuals)
    pred_high = all_preds > median_revenue
    actual_high = all_actuals > median_revenue
    
    accuracy = np.mean(pred_high == actual_high) * 100
    
    if np.sum(pred_high) > 0:
        precision = np.sum(pred_high & actual_high) / np.sum(pred_high) * 100
    else:
        precision = 0
        
    if np.sum(actual_high) > 0:
        recall = np.sum(pred_high & actual_high) / np.sum(actual_high) * 100
    else:
        recall = 0
        
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    test_metrics = {
        'test_loss': np.mean(test_losses),
        'within_25': within_25,
        'within_50': within_50,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    print("\nTest Set Metrics:")
    print("-" * 50)
    print(f"Test Loss:     {test_metrics['test_loss']:.4f}")
    print(f"Within 25%:    {test_metrics['within_25']:.1f}%")
    print(f"Within 50%:    {test_metrics['within_50']:.1f}%")
    print(f"Accuracy:      {test_metrics['accuracy']:.1f}%")
    print(f"Precision:     {test_metrics['precision']:.1f}%")
    print(f"Recall:        {test_metrics['recall']:.1f}%")
    print(f"F1 Score:      {test_metrics['f1_score']:.1f}")
    print("-" * 50)
    
    return test_metrics

def single_run(train_loader, val_loader, test_loader):
    params = {
        'learning_rate': 1e-4,      # Smaller, stable steps
        'weight_decay': 0.001,      # Gentle regularization
        'epochs': 20,               # Same
        'patience': 5,              # Same
        'l1_lambda': 0,            # Skip L1 regularization
        'clip_grad': 1.0,          # Keep grad clipping for stability
        'batch_size': 32           # Larger batch for stability
    }
    
    print("\nRunning with carefully chosen parameters:")
    print(json.dumps(params, indent=2))
    
    val_loss = train_with_params(train_loader, val_loader, params)
    
    # Load best model for test evaluation
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = RevenuePredictor().to(device)
    model.load_state_dict(torch.load(f'newmodels/{MODEL_NAME}.pth'))
    
    # Get test metrics
    test_metrics = evaluate_on_test(model, test_loader, device)
    
    # Save all results
    with open(f'newmodels/{MODEL_NAME}.json', 'w') as f:
        json.dump({
            'val_loss': val_loss,
            'params': params,
            'test_metrics': test_metrics
        }, f, indent=4)
    
    return val_loss, test_metrics

def main():
    os.makedirs('newmodels', exist_ok=True)
    
    print("Loading clean dataset...")
    dataset = load_from_disk(DATASET_NAME)
    
    print(f"Splitting into train/val/test... for model saving name: {MODEL_NAME}")
    
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_val = splits["train"]
    test_dataset = splits["test"]
    
    train_val_splits = train_val.train_test_split(test_size=0.1, seed=42)
    
    # Create datasets
    train_dataset = MoviePosterDataset(train_val_splits, split="train")
    val_dataset = MoviePosterDataset(train_val_splits, split="test")
    test_dataset = MoviePosterDataset({"train": test_dataset}, split="train")
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print(f"\nDataset splits:")
    print(f"Training:   {len(train_dataset):,d} examples")
    print(f"Validation: {len(val_dataset):,d} examples")
    print(f"Test:       {len(test_dataset):,d} examples")
    
    # Run single training with best params
    val_loss, test_metrics = single_run(train_loader, val_loader, test_loader)
    
    # Print final results
    print("\nFinal Results:")
    print("=" * 50)
    print(f"Best Validation Loss: {val_loss:.4f}")
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            if 'loss' in metric.lower():
                print(f"{metric:15}: {value:.4f}")
            else:
                print(f"{metric:15}: {value:.1f}%")
    print("=" * 50)
    
    # Save final results
    results = {
        'validation_loss': val_loss,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f'newmodels/final_results_{MODEL_NAME}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: newmodels/final_results_{MODEL_NAME}.json")

if __name__ == "__main__":
    main()