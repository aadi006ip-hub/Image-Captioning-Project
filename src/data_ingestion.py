import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, caption_file_path, image_dir):
        self.caption_file_path = caption_file_path
        self.image_dir = image_dir

    def load_data(self) -> pd.DataFrame:
        # File read karo
        df = pd.read_csv(self.caption_file_path)
        
        # AGAR column ka naam 'image' hai, toh usko 'image_id' bana do
        if 'image' in df.columns:
            df = df.rename(columns={'image': 'image_id'})
            
        # Ab check karo ki image_path bana ya nahi
        if 'image_id' in df.columns:
            df['image_path'] = df['image_id'].apply(lambda x: os.path.join(self.image_dir, x))
            
        return df

    def split_data(self, df: pd.DataFrame, test_size=0.2, random_state=42):
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)