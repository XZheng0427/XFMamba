import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd

class HipXrayDataset(Dataset):
    def __init__(self, images_dir, csv_file, transform=None):
        """
        Initializes the HipXrayDataset.

        Args:
            images_dir (str): Path to the directory containing the images.
            csv_file (str): Path to the CSV file containing the dataset information.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., data augmentation, normalization).
        """
        self.images_dir = images_dir
        self.transform = transform

        try:
            self.data_table = pd.read_csv(csv_file)
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file: {e}")

        # Validate required columns
        required_columns = ['Image_no', 'Admission_Date', 'Fracture_Classification', 'Final_Classification']
        if not all(column in self.data_table.columns for column in required_columns):
            raise ValueError(f"CSV file must contain the following columns: {required_columns}")

        # Exclude entries where 'Fracture_Classification' is 'Exclude'
        self.data_table = self.data_table[self.data_table['Fracture_Classification'].str.lower() != 'exclude']

        # Initialize a list to hold valid image pairs and their labels
        self.valid_data = []

        # Iterate over each row in the filtered DataFrame
        for _, row in self.data_table.iterrows():
            img_no = row['Image_no']
            img_no_str = str(img_no).zfill(3)  # Zero-pad image number to 3 digits

            # Define paths for AP and Lateral images
            ap_image_path = os.path.join(images_dir, f"{img_no_str} AP.jpg")
            lateral_image_path = os.path.join(images_dir, f"{img_no_str} Lateral.jpg")

            # Check if both AP and Lateral images exist
            if os.path.exists(ap_image_path) and os.path.exists(lateral_image_path):
                # Ensure 'Final_Classification' is not NaN
                if pd.notna(row['Final_Classification']):
                    self.valid_data.append({
                        'ap_path': ap_image_path,
                        'lateral_path': lateral_image_path,
                        'label': row['Final_Classification']
                    })
                else:
                    print(f"Skipping Image {img_no_str}: 'Final Classification' is missing.")
            else:
                print(f"Skipping Image {img_no_str}: Missing AP or Lateral view.")

        # If no valid data is found, raise an error
        if not self.valid_data:
            raise RuntimeError("No valid image pairs found. Please check the image directory and CSV file.")

        # Create a mapping from label names to integer indices
        unique_labels = sorted(self.data_table['Final_Classification'].dropna().unique())
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        print("Label Mapping:", self.label_mapping)

        # Apply the mapping to the labels in valid_data
        for entry in self.valid_data:
            entry['label'] = self.label_mapping[entry['label']]

    def __len__(self):
        """Returns the total number of valid image pairs."""
        return len(self.valid_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.valid_data[idx]

        # Load images
        ap_image = Image.open(sample['ap_path']).convert('L')
        lateral_image = Image.open(sample['lateral_path']).convert('L')

        # Apply transformations if any
        if self.transform:
            ap_image = self.transform(ap_image)
            lateral_image = self.transform(lateral_image)

        label = sample['label']

        return {'ap': ap_image, 'lateral': lateral_image, 'label': label}

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.subset[index]
        ap_image = sample['ap']
        lateral_image = sample['lateral']
        label = sample['label']
        
        if self.transform:
            ap_image = self.transform(ap_image)
            lateral_image = self.transform(lateral_image)
        
        return ap_image, lateral_image, label

    def __len__(self):
        return len(self.subset)

def create_data_loader(images_dir, csv_file, config):
    train_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Since images are grayscale
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Since images are grayscale
    ])

    train_dataset = HipXrayDataset(
        images_dir=images_dir,
        csv_file=csv_file,
        transform=None
    )

    total_size = len(train_dataset)
    train_size = int(config['train_ratio'] * total_size)
    valid_size = total_size - train_size  # Ensures that all samples are used

    train_subset, valid_subset = random_split(
        train_dataset, 
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(config['seed'])  # Ensures reproducibility
    )
    train_dataset = SubsetWithTransform(train_subset, transform=train_transform)
    val_dataset = SubsetWithTransform(valid_subset, transform=val_transform)

    print(f'Train_dataset: {len(train_dataset)} Val_dataset: {len(val_dataset)}')

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=config['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory']
    )

    return train_loader, val_loader

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    images_dir = '/data/DERI-USMSK/XavierHipXray/Images'
    csv_file = '/data/DERI-USMSK/XavierHipXray/hipxray-label.csv'

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Since images are grayscale
    ])

    train_dataset = HipXrayDataset(
        images_dir=images_dir,
        csv_file=csv_file,
        transform=train_transform
    )

    print(f'Train_dataset: {len(train_dataset)}')