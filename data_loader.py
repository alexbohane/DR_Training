# data_loader.py

import pandas as pd
import torchvision
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset_class import MessidorOpenCVDataset
from preprocess_class import OpenCV_DR_Preprocessor
from transforms import light_transform, heavy_transform, test_transform


# --- Initialize Preprocessor ---
preprocessor = OpenCV_DR_Preprocessor(apply_clahe=True, apply_roi_mask=True)

#define root dir
root_dir = '/workspace/DR_Training/MESSIDOR'
# root_dir='/Users/abohane/Desktop/THEIA Training/MESSIDOR'

# --- Load Full Dataset (for dataframe) ---
full_dataset = MessidorOpenCVDataset(
    root_dir=root_dir,
    preprocessor=preprocessor,
    light_transform=None,
    heavy_transform=None,
    final_transform=False,  # No final transform for full datas
    minority_classes=[1]
)

df = full_dataset.data  # Full annotations dataframe

# --- Train/Test Split ---
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['Retinopathy grade'], # type: ignore
    random_state=42
)

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# --- Create Train and Test Datasets ---
train_dataset = MessidorOpenCVDataset(
    root_dir=root_dir,
    preprocessor=preprocessor,
    light_transform=light_transform,
    heavy_transform=heavy_transform,
    final_transform=False,  # Apply final transform for training
    minority_classes=[1]
)
train_dataset.data = train_df.reset_index(drop=True) # type: ignore

test_dataset = MessidorOpenCVDataset(
    root_dir=root_dir,
    preprocessor=preprocessor,
    light_transform=test_transform,
    heavy_transform=test_transform,
    final_transform=False,  # Apply final transform for testing
    minority_classes=[1]
)
test_dataset.data = test_df.reset_index(drop=True) # type: ignore

# --- Create DataLoaders ---
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)