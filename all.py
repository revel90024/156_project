import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from datasets import load_from_disk
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
from datetime import datetime
import os
from tqdm import tqdm
from contextlib import nullcontext

class MovieAllFeaturesDataset(Dataset):
    def __init__(self, hf_dataset, split="train", transform=None):
        self.dataset = hf_dataset[split]
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        
        # Compute mean and std for revenue and budget in log space
        self.revenue_logs = np.array([np.log1p(float(item['revenue'])) for item in self.dataset])
        self.budget_logs = np.array([np.log1p(float(item['budget'])) for item in self.dataset])
        
        self.revenue_mean = self.revenue_logs.mean()
        self.revenue_std = self.revenue_logs.std()
        self.budget_mean = self.budget_logs.mean()
        self.budget_std = self.budget_logs.std()
        
        # Similarly for runtime
        runtimes = np.array([float(item.get('runtime', 120)) for item in self.dataset])
        self.runtime_mean = runtimes.mean()
        self.runtime_std = runtimes.std()
        
        # For release dates
        release_dates = np.array([
            (datetime.strptime(item['release_date'], '%Y-%m-%d') - datetime(2000, 1, 1)).days
            for item in self.dataset
        ])
        self.release_mean = release_dates.mean()
        self.release_std = release_dates.std()
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. Image processing
        image = self.transform(item['image'])
        
        # 2. Numerical features - standardization
        budget = np.log1p(float(item['budget']))
        budget = (budget - self.budget_mean) / self.budget_std
        
        runtime = float(item.get('runtime', self.runtime_mean))
        runtime = (runtime - self.runtime_mean) / self.runtime_std
        
        release_date = datetime.strptime(item['release_date'], '%Y-%m-%d')
        days_since_2000 = (release_date - datetime(2000, 1, 1)).days
        days_since_2000 = (days_since_2000 - self.release_mean) / self.release_std
        
        numerical_features = torch.tensor([
            budget,
            runtime,
            days_since_2000
        ], dtype=torch.float32)
        
        # 3. Text features
        text = f"{item.get('title', '')} {item.get('tagline', '')} {item.get('overview', '')}"
        text_features = self.tokenizer(
            text,
            padding='max_length',
            max_length=256,  # Reduce max_length to save memory
            truncation=True,
            return_tensors='pt'
        )
        
        # 4. Genre features
        genres = json.loads(item.get('genres', '[]')) if isinstance(item.get('genres'), str) else item.get('genres', [])
        genre_vector = torch.zeros(20)
        for genre in genres:
            if isinstance(genre, dict) and 'id' in genre:
                genre_id = genre['id'] % 20  # Simplified for this example
                genre_vector[genre_id] = 1
        
        # 5. Target - standardize revenue
        revenue = np.log1p(float(item['revenue']))
        revenue = (revenue - self.revenue_mean) / self.revenue_std
        target = torch.tensor(revenue, dtype=torch.float32)
        
        return {
            'image': image,
            'numerical': numerical_features,
            'text_ids': text_features['input_ids'].squeeze(0),
            'text_mask': text_features['attention_mask'].squeeze(0),
            'genres': genre_vector,
            'target': target
        }

class MultiModalRevenuePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Image backbone with partially frozen layers
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for name, param in self.resnet.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        self.resnet.fc = nn.Identity()
        
        # 2. Smaller BERT with partially frozen layers
        self.bert = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        for name, param in self.bert.named_parameters():
            if 'encoder.layer.1' not in name and 'pooler' not in name:
                param.requires_grad = False
                
        # 3. Fusion network with increased capacity
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 128 + 3 + 20, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # Initialize weights properly
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, batch):
        with torch.cuda.amp.autocast() if torch.cuda.is_available() or torch.backends.mps.is_available() else nullcontext():
            img_features = self.resnet(batch['image'])
            
            text_outputs = self.bert(
                input_ids=batch['text_ids'],
                attention_mask=batch['text_mask']
            )
            text_features = text_outputs.last_hidden_state[:, 0, :]
            
            combined = torch.cat([
                img_features,
                text_features,
                batch['numerical'],
                batch['genres']
            ], dim=1)
            
            return self.fusion(combined)

def calculate_metrics(predictions, actuals, threshold=0.5):
    """Calculate various regression and classification metrics"""
    predictions = np.array(predictions).squeeze()
    actuals = np.array(actuals).squeeze()
    
    # Convert back from standardized log space to original revenue
    predictions = np.expm1(predictions * revenue_std + revenue_mean)
    actuals = np.expm1(actuals * revenue_std + revenue_mean)
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    # Regression metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # Error percentages
    errors = np.abs(predictions - actuals) / (actuals + epsilon) * 100
    within_25 = np.mean(errors <= 25) * 100
    within_50 = np.mean(errors <= 50) * 100
    
    # Revenue brackets for classification metrics
    median_revenue = np.median(actuals)
    pred_high = predictions > median_revenue
    actual_high = actuals > median_revenue
    
    accuracy = np.mean(pred_high == actual_high) * 100
    
    # Precision and recall for high revenue predictions
    true_pos = np.sum(pred_high & actual_high)
    false_pos = np.sum(pred_high & ~actual_high)
    false_neg = np.sum(~pred_high & actual_high)
    
    precision = true_pos / (true_pos + false_pos + epsilon) * 100
    recall = true_pos / (true_pos + false_neg + epsilon) * 100
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Within 25%': within_25,
        'Within 50%': within_50,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

def evaluate_model(model, val_loader, device, revenue_mean, revenue_std):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(batch).squeeze()
            loss = nn.MSELoss()(outputs, batch['target'])
            total_loss += loss.item()
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch['target'].cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    metrics = calculate_metrics(predictions, actuals)
    
    print("\n" + "="*80)
    print(f"EVALUATION METRICS:")
    print("="*80)
    print(f"Loss:            {avg_loss:.4f}")
    print(f"RMSE:            ${metrics['RMSE']:,.2f}")
    print(f"MAE:             ${metrics['MAE']:,.2f}")
    print(f"Within 25%:      {metrics['Within 25%']:.1f}%")
    print(f"Within 50%:      {metrics['Within 50%']:.1f}%")
    print(f"Accuracy:        {metrics['Accuracy']:.1f}%")
    print(f"Precision:       {metrics['Precision']:.1f}%")
    print(f"Recall:          {metrics['Recall']:.1f}%")
    print(f"F1 Score:        {metrics['F1']:.1f}%")
    print("="*80)
    
    return avg_loss, metrics

def train_model():
    # Setup
    os.makedirs('models', exist_ok=True)
    
    # Adjusted device selection to include MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    
    # Training parameters
    EPOCHS = 20
    BATCH_SIZE = 16  # Increased batch size
    LEARNING_RATE = 1e-4  # Adjusted learning rate
    PATIENCE = 5
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk("clean_movies_1M_modern")
    
    # Filter out entries with zero revenue or budget
    dataset = dataset.filter(lambda x: float(x['revenue']) > 0 and float(x['budget']) > 0)
    
    # Split dataset
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_val = splits["train"].train_test_split(test_size=0.1, seed=42)
    
    # Create dataloaders
    train_dataset = MovieAllFeaturesDataset(train_val, split="train")
    val_dataset = MovieAllFeaturesDataset(train_val, split="test")
    test_dataset = MovieAllFeaturesDataset({"train": splits["test"]}, split="train")
    
    global revenue_mean, revenue_std
    revenue_mean = train_dataset.revenue_mean
    revenue_std = train_dataset.revenue_std
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"\nDataset splits:")
    print(f"Training:   {len(train_dataset):,d} examples")
    print(f"Validation: {len(val_dataset):,d} examples")
    print(f"Test:       {len(test_dataset):,d} examples")
    
    # Initialize model and optimizer
    model = MultiModalRevenuePredictor().to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Use Huber Loss for robustness to outliers
    loss_fn = nn.SmoothL1Loss()
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = None
    early_stopping_count = 0
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() or torch.backends.mps.is_available() else None
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-"*80)
        
        # Training progress
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(batch).squeeze()
                    loss = loss_fn(outputs, batch['target'])
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch).squeeze()
                loss = loss_fn(outputs, batch['target'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar with current loss
            progress_bar.set_postfix({
                'loss': f"{total_loss/batch_count:.4f}"
            })
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        print("\nValidating...")
        val_loss, val_metrics = evaluate_model(model, val_loader, device, revenue_mean, revenue_std)
        
        # Learning rate adjustment
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"\nLearning rate adjusted: {old_lr:.2e} -> {new_lr:.2e}")
        
        # Model saving and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': val_metrics
            }, 'models/best_all_features.pth')
            print("\nSaved new best model!")
            early_stopping_count = 0
        else:
            early_stopping_count += 1
            if early_stopping_count >= PATIENCE:
                print("\nEarly stopping triggered!")
                break
    
    # Print best results
    print("\nTraining completed!")
    print("="*80)
    print("Best Model Metrics:")
    print("="*80)
    for metric, value in best_metrics.items():
        if 'Within' in metric or 'Accuracy' in metric or 'Precision' in metric or 'Recall' in metric or 'F1' in metric:
            print(f"{metric+':':<15} {value:.2f}%")
        else:
            print(f"{metric+':':<15} ${value:,.2f}")
    print("="*80)
    
    # Final evaluation on test set
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load('models/best_all_features.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_metrics = evaluate_model(model, test_loader, device, revenue_mean, revenue_std)
    
if __name__ == "__main__":
    train_model()
