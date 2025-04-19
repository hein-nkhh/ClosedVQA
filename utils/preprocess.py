import re
from PIL import Image
from torchvision import transforms

def clean_text(text):
    text = re.sub(r'\([^)]*\)', '', text)  # Remove parentheses
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

image_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # <-- phải trước
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # <-- sau
])