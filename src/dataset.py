import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from src.data_preprocessing import ImagePreprocessor

class ImageCaptionDataset(Dataset):
    def __init__(self, df, vocab, transform_size=(224, 224)):
        self.df = df
        self.vocab = vocab
        self.image_preprocessor = ImagePreprocessor(image_size=transform_size)

    def __len__(self): return len(self.df)

    def __getitem__(self, index):
        caption = self.df.iloc[index]['caption']
        image_path = self.df.iloc[index]['image_path']
        return self.image_preprocessor.preprocess_image(image_path), torch.tensor(self.vocab.numericalize(caption))

class CapsCollate:
    def __init__(self, pad_idx): self.pad_idx = pad_idx
    def __call__(self, batch):
        images = torch.cat([item[0].unsqueeze(0) for item in batch], dim=0)
        targets = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=self.pad_idx)
        return images, targets