import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from PIL import Image

class PretrainModel(nn.Module):
    def __init__(self, device=None):
        super(PretrainModel, self).__init__()
        self.image_encoder = AutoModel.from_pretrained("bertin-project/skincon").to(device)
        self.image_processor = AutoImageProcessor.from_pretrained("bertin-project/skincon")
        
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
        self.text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
        
        self.projection_dim = 256
        self.hidden_dim = 768
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_encoder.config.hidden_size, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )

    def forward(self, text_inputs, image_inputs, device=None):
        image_inputs = self.image_processor(images=image_inputs, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

        image_features = self.image_encoder(**image_inputs).last_hidden_state.mean(dim=1)
        image_embeddings = self.image_proj(image_features)
        
        text_inputs = self.text_tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = self.text_encoder(**text_inputs).last_hidden_state[:, 0, :]
        text_embeddings = self.text_proj(text_features)
        
        image_embeddings = nn.functional.normalize(image_embeddings, dim=-1)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)
        
        return image_embeddings, text_embeddings