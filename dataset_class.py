import os
import pandas as pd
from torch.utils.data import Dataset
import glob
import torch

class MessidorOpenCVDataset(Dataset):
    def __init__(self, root_dir, preprocessor, light_transform=None, heavy_transform=None, final_transform=False, minority_classes=[1]):
        """
        Args:
            root_dir (str): Path to the root directory containing folders and xls files
            preprocessor (OpenCV_DR_Preprocessor): Preprocessing class
        """
        self.root_dir = root_dir
        self.preprocessor = preprocessor
        self.light_transform = light_transform
        self.heavy_transform = heavy_transform
        self.minority_classes = minority_classes
        self.data = self._load_all_annotations() # This is the dataset
        self.final_transform = final_transform

    def _load_all_annotations(self):
        """Loads all .xls annotation files and creates a master dataframe."""
        all_data = []

        # Find all .xls files
        annotation_files = glob.glob(os.path.join(self.root_dir, "*.xls"))

        for annotation_file in annotation_files:
            # Example: "Annotation_Base11.xls" -> folder is "Base11"
            folder_name = os.path.splitext(os.path.basename(annotation_file))[0].replace('Annotation_', '')

            df = pd.read_excel(annotation_file)

            # Extract relevant columns
            df = df[['Image name', 'Retinopathy grade']]
            df['folder'] = folder_name  # Add the folder name

            all_data.append(df)

        # Concatenate all into one big dataframe
        full_df = pd.concat(all_data, ignore_index=True)

        return full_df

    def _convert_to_binary_label(self, original_label):
        """Convert 4-class labels to binary: 0,1 -> 0 (healthy), 2,3 -> 1 (unhealthy)"""
        if original_label in [0, 1]:
            return 0  # Healthy
        else:
            return 1  # Unhealthy

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        """ Finds the image associated with df that has been created by looking for file name"""
        row = self.data.iloc[idx]
        img_name = row['Image name']
        original_label = row['Retinopathy grade']
        folder = row['folder']

        # Convert to binary label
        binary_label = self._convert_to_binary_label(original_label)

        # Full path to image
        img_path = os.path.join(self.root_dir, folder, img_name)

        # Preprocess image
        img = self.preprocessor.preprocess(img_path, normalize=False)[0]
        label = torch.tensor(binary_label, dtype=torch.long)

        # Apply transformations based on class
        if label.item() in self.minority_classes:
            if self.heavy_transform:
                img = self.heavy_transform(img)
        else:
            if self.light_transform:
                img = self.light_transform(img)

        img = self.preprocessor.normalize(img)

        return img, label
