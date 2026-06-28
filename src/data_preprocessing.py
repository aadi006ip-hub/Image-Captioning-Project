import string
import re
from PIL import Image
import torchvision.transforms as transforms

class ImagePreprocessor:
    def __init__(self, image_size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

class TextPreprocessor:
    @staticmethod
    def clean_caption(caption: str) -> str:
        caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
        caption = re.sub(r'\s+', ' ', caption).strip()
        return f"startseq {caption} endseq"