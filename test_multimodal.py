import torch
from torchvision import transforms
from PIL import Image
import json
from adi_tester import MultiModalModel, MoviePosterDataset
from transformers import BertTokenizer

def load_model(model_path, num_genres, num_languages):
    model = MultiModalModel(
        num_genres=num_genres,
        num_languages=num_languages,
        hidden_sizes=[1024, 512, 1024],
        dropout_rates=[0.2, 0.1, 0.2]
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, device

def predict_revenue(image_path, title, overview, budget, runtime, genres, language, release_date):
    # Load model and tokenizer
    model, device = load_model('adi_large_50M_multimodal/adi_dec4_50M_multimodal.pth')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Process image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Process text
    text = f"{title} {overview}"
    text_inputs = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=512
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    # Process numerical features
    # (Add your feature processing here similar to MoviePosterDataset)
    
    with torch.no_grad():
        output = model(image_tensor, text_inputs, numerical_features)
        predicted_revenue = float(10 ** output.squeeze().cpu().numpy())
    
    return predicted_revenue 