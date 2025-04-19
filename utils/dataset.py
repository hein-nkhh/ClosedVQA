import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from .preprocess import clean_text, image_transform

class PretrainDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = [json.loads(line) for line in f]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        caption = clean_text(item["caption"])
        return image_transform(image), caption

class VQADataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        question = clean_text(item["question"])
        answers = [clean_text(ans) for ans in item["answer_list"]]
        label = item["correct_idx"]
        return image_transform(image), question, answers, label