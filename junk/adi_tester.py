MODEL_NAME = "adi_dec4_50M_multimodal"
DATASET_NAME = "large_50M_torch"
FOLDER_NAME = "adi_large_50M_multimodal"
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4

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
from transformers import BertTokenizer, BertModel
from collections import defaultdict

def get_all_genres(dataset):
    """Get unique genres from dataset"""
    genres = set()
    for item in dataset:
        for genre in item['genres']:
            genres.add(genre['name'])
    return sorted(list(genres))

def get_all_languages(dataset):
    """Get unique languages from dataset"""
    return sorted(list(set(item['original_language'] for item in dataset)))

class MoviePosterDataset(Dataset):
    def __init__(self, hf_dataset, split="train", transform=None, tokenizer=None, genres=None, languages=None):
        self.dataset = hf_dataset[split]
        self.transform = transform or transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        
        # Use provided genres/languages or get from dataset
        if genres is None or languages is None:
            print("Getting unique genres and languages...")
            self.genres = get_all_genres(self.dataset)
            self.languages = get_all_languages(self.dataset)
        else:
            self.genres = genres
            self.languages = languages
        
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
        self.lang_to_idx = {lang: idx for idx, lang in enumerate(self.languages)}
        
        print(f"Found {len(self.genres)} unique genres and {len(self.languages)} languages")

    def __len__(self):
        return len(self.dataset)
    
    def process_date(self, date_str):
        """Convert date string to year and month features"""
        try:
            year = int(date_str[:4])
            month = int(date_str[5:7])
            # Normalize
            year = (year - 1900) / 150  # Assuming movies from 1900-2050
            month = month / 12
            return [year, month]
        except:
            return [0.0, 0.0]
    
    def one_hot_encode(self, genres, num_genres):
        """Convert genres list to one-hot encoding"""
        encoding = np.zeros(num_genres)
        for genre in genres:
            if genre['name'] in self.genre_to_idx:
                encoding[self.genre_to_idx[genre['name']]] = 1
        return encoding

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. Process image
        image = torch.tensor(item['image'])
        image = self.transform(image)
        
        # 2. Process text data
        title = item.get('title', '')
        tagline = item.get('tagline', '')
        overview = item.get('overview', '')
        text = f"{title} {tagline} {overview}"
        text_inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=512
        )
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
        # 3. Process numerical features
        budget = np.log1p(float(item.get('budget', 0))) / 25
        runtime = float(item.get('runtime', 0)) / 300
        
        # Genre one-hot encoding
        genre_encoding = self.one_hot_encode(item.get('genres', []), len(self.genres))
        
        # Language one-hot encoding
        lang_encoding = np.zeros(len(self.languages))
        lang_idx = self.lang_to_idx.get(item.get('original_language', 'en'), 0)
        lang_encoding[lang_idx] = 1
        
        # Date features
        date_features = self.process_date(item.get('release_date', '2000-01-01'))
        
        # Combine all numerical features
        numerical_features = np.concatenate([
            [budget, runtime],
            genre_encoding,
            lang_encoding,
            date_features
        ])
        
        # 4. Get revenue (target)
        revenue = float(item['revenue'])
        if revenue > 0:
            revenue = np.log10(revenue)
        else:
            revenue = 0.0
        
        return (
            image,
            text_inputs,
            torch.tensor(numerical_features, dtype=torch.float32),
            torch.tensor(revenue, dtype=torch.float32)
        )

class MultiModalModel(nn.Module):
    def __init__(self, num_genres, num_languages, hidden_sizes, dropout_rates):
        super().__init__()
        # 1. Image model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_image_features = self.resnet.fc.in_features  # Should be 2048
        self.resnet.fc = nn.Identity()
        
        # 2. Text model
        self.bert = BertModel.from_pretrained('prajjwal1/bert-tiny')
        num_text_features = self.bert.config.hidden_size  # Should be 128 for bert-tiny
        
        # 3. Calculate numerical features size
        num_numerical_features = (
            2 +  # budget and runtime
            num_genres +  # one-hot encoded genres
            num_languages +  # one-hot encoded languages
            2  # year and month from release date
        )
        
        # Print feature dimensions for debugging
        print(f"Feature dimensions:")
        print(f"Image features: {num_image_features}")
        print(f"Text features: {num_text_features}")
        print(f"Numerical features: {num_numerical_features}")
        
        # 4. Combined features
        self.total_features = num_image_features + num_text_features + num_numerical_features
        print(f"Total combined features: {self.total_features}")
        
        # 5. Build combined layers
        layers = []
        in_features = self.total_features  # Store this for reference
        
        for hidden_size, dropout_rate in zip(hidden_sizes, dropout_rates):
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Add BatchNorm
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size
        
        layers.append(nn.Linear(in_features, 1))
        self.combined_fc = nn.Sequential(*layers)
    
    def forward(self, image, text_inputs, numerical_data):
        # 1. Process image
        image_features = self.resnet(image)  # [batch_size, 2048]
        
        # 2. Process text
        text_outputs = self.bert(**text_inputs)
        text_features = text_outputs.pooler_output  # [batch_size, 128]
        
        # 3. Combine features
        combined_features = torch.cat([
            image_features,
            text_features,
            numerical_data
        ], dim=1)
        
        # Debug dimensions
        if combined_features.shape[1] != self.total_features:
            raise ValueError(f"Expected {self.total_features} features but got {combined_features.shape[1]}")
        
        # 4. Final prediction
        return self.combined_fc(combined_features)

def calculate_metrics(outputs, revenues):
    """Calculate all metrics for the epoch"""
    pred_revenue = 10 ** outputs.squeeze()
    actual_revenue = 10 ** revenues
    
    # Handle any NaN values
    mask = ~np.isnan(pred_revenue) & ~np.isnan(actual_revenue)
    pred_revenue = pred_revenue[mask]
    actual_revenue = actual_revenue[mask]
    
    if len(pred_revenue) == 0:
        return {
            'within_25': 0,
            'within_50': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }
    
    # Calculate percentage errors
    errors = np.abs(pred_revenue - actual_revenue) / actual_revenue
    within_25 = np.mean(errors <= 0.25) * 100
    within_50 = np.mean(errors <= 0.50) * 100
    
    # Calculate classification metrics
    median_revenue = np.median(actual_revenue)
    pred_high = pred_revenue > median_revenue
    actual_high = actual_revenue > median_revenue
    
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
    
    return {
        'within_25': within_25,
        'within_50': within_50,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def train_with_params(train_loader, val_loader, params):
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Get number of genres and languages from the dataset
    num_genres = len(train_loader.dataset.genres)
    num_languages = len(train_loader.dataset.languages)
    
    # Create model
    model = MultiModalModel(
        num_genres=num_genres,
        num_languages=num_languages,
        hidden_sizes=params['hidden_sizes'],
        dropout_rates=params['dropout_rates']
    )
    
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
        all_train_outputs = []
        all_train_revenues = []
        
        for i, (images, text_inputs, numerical_data, revenues) in enumerate(train_loader):
            images = images.to(device)
            numerical_data = numerical_data.to(device)
            revenues = revenues.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            outputs = model(images, text_inputs, numerical_data)
            loss = nn.MSELoss()(outputs, revenues.unsqueeze(1))
            
            # Normalize loss for gradient accumulation
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            # Step optimizer only after accumulating gradients
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_losses.append(loss.item() * ACCUMULATION_STEPS)  # Denormalize for logging
            
            # Store predictions for metrics
            all_train_outputs.extend(outputs.detach().cpu().numpy())
            all_train_revenues.extend(revenues.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_losses = []
        all_val_outputs = []
        all_val_revenues = []
        
        with torch.no_grad():
            for images, text_inputs, numerical_data, revenues in val_loader:
                images = images.to(device)
                numerical_data = numerical_data.to(device)
                revenues = revenues.to(device)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                
                outputs = model(images, text_inputs, numerical_data)
                val_loss = nn.MSELoss()(outputs, revenues.unsqueeze(1))
                val_losses.append(val_loss.item())
                
                # Store predictions for metrics
                all_val_outputs.extend(outputs.cpu().numpy())
                all_val_revenues.extend(revenues.cpu().numpy())
        
        # Calculate metrics
        train_metrics = calculate_metrics(
            np.array(all_train_outputs),
            np.array(all_train_revenues)
        )
        val_metrics = calculate_metrics(
            np.array(all_val_outputs),
            np.array(all_val_revenues)
        )
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        print(f"\nEpoch {epoch+1}/{params['epochs']}")
        print("-" * 50)
        print(f"Train Loss:     {avg_train_loss:.4f}")
        print(f"Val Loss:       {avg_val_loss:.4f}")
        print("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"{metric:12}: {value:.1f}%")
        print("\nValidation Metrics:")
        for metric, value in val_metrics.items():
            print(f"{metric:12}: {value:.1f}%")
        print("-" * 50)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{FOLDER_NAME}/{MODEL_NAME}.pth')
            early_stopping_count = 0
        else:
            early_stopping_count += 1
            if early_stopping_count >= params['patience']:
                print("\nEarly stopping triggered!")
                break
    
    return best_val_loss

def grid_search(train_loader, val_loader):
    param_grid = {
        'learning_rate': [1e-3, 5e-4, 1e-4, 3e-3], # experiment with making learning rate a little bigger 
        'weight_decay': [0.01, 0.001, 0.005],  
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
        with open(f'{FOLDER_NAME}/{MODEL_NAME}.json', 'w') as f:
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
            pred_revenue = 10 ** outputs.squeeze()
            actual_revenue = 10 ** revenues
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
    # Get number of genres and languages from the dataset
    num_genres = len(train_loader.dataset.genres)
    num_languages = len(train_loader.dataset.languages)
    
    params = {
        # Training params
        'learning_rate': 2e-4,      # Higher learning rate
        'weight_decay': 0.0001,     # Less regularization
        'epochs': 30,               # even more epochs
        'patience': 7,              # More patience
        'l1_lambda': 0,            # No L1
        'clip_grad': None,         # No gradient clipping
        'batch_size': BATCH_SIZE,  # Use the constant instead of hardcoding
        
        # Model architecture params
        'hidden_sizes': [1024, 512, 1024],  # try 3 hidden layers  
        'dropout_rates': [0.2, 0.1, 0.2],   # lower the dropout
        
        # Add these to params
        'num_genres': num_genres,
        'num_languages': num_languages
    }
    
    print("\nRunning with less regularization:")
    print(json.dumps(params, indent=2))
    
    # Create model with specified architecture
    model = MultiModalModel(
        num_genres=params['num_genres'],
        num_languages=params['num_languages'],
        hidden_sizes=params['hidden_sizes'],
        dropout_rates=params['dropout_rates']
    )
    
    val_loss = train_with_params(train_loader, val_loader, params)
    
    # Load best model for test evaluation
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiModalModel(
        num_genres=params['num_genres'],
        num_languages=params['num_languages'],
        hidden_sizes=params['hidden_sizes'],
        dropout_rates=params['dropout_rates']
    ).to(device)
    model.load_state_dict(torch.load(f'{FOLDER_NAME}/{MODEL_NAME}.pth'))
    
    # Get test metrics
    test_metrics = evaluate_on_test(model, test_loader, device)
    
    # Save all results
    with open(f'{FOLDER_NAME}/{MODEL_NAME}.json', 'w') as f:
        json.dump({
            'val_loss': val_loss,
            'params': params,
            'test_metrics': test_metrics
        }, f, indent=4)
    
    return val_loss, test_metrics

def main():
    os.makedirs(FOLDER_NAME, exist_ok=True)
    
    print("Loading clean dataset...")
    dataset = load_from_disk(DATASET_NAME)
    
    # Get genres and languages from FULL dataset before splitting
    all_genres = get_all_genres(dataset)
    all_languages = get_all_languages(dataset)
    print(f"Found {len(all_genres)} genres and {len(all_languages)} languages in full dataset")
    
    print(f"Splitting into train/val/test... for model saving name: {MODEL_NAME}")
    
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_val = splits["train"]
    test_dataset = splits["test"]
    
    train_val_splits = train_val.train_test_split(test_size=0.1, seed=42)
    
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets with shared genre/language mappings
    train_dataset = MoviePosterDataset(
        train_val_splits, 
        split="train", 
        tokenizer=tokenizer,
        genres=all_genres,
        languages=all_languages
    )
    val_dataset = MoviePosterDataset(
        train_val_splits, 
        split="test", 
        tokenizer=tokenizer,
        genres=all_genres,
        languages=all_languages
    )
    test_dataset = MoviePosterDataset(
        {"train": test_dataset}, 
        split="train", 
        tokenizer=tokenizer,
        genres=all_genres,
        languages=all_languages
    )
    
    # Create loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
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
    
    with open(f'{FOLDER_NAME}/final_results_{MODEL_NAME}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {FOLDER_NAME}/final_results_{MODEL_NAME}.json")

if __name__ == "__main__":
    main()