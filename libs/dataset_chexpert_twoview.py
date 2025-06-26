import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class CheXpertDataset(Dataset):
    LABELS = (
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
        'Support Devices', 'No Finding',
    )

    def __init__(self, csv_files, root_dirs, transform=None, subset_indices=None, verbose=False):
        """
        Args:
            csv_files (list of str): List of paths to CSV annotation files.
            root_dirs (list of str): List of root directories where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            subset_indices (list or None): List of indices to include in the dataset subset.
            verbose (bool): If True, prints detailed logs.
        """
        self.root_dirs = root_dirs
        self.transform = transform
        self.verbose = verbose
        self.samples = []  # Each sample is (frontal_path, lateral_path, labels_tensor)
        self.image_path_map = {}  # Mapping from relative path to full path

        if self.verbose:
            print("Initializing CheXpertDataset...")
            print(f"CSV files: {csv_files}")
            print(f"Root directories: {self.root_dirs}")

        # Build the image path mapping
        self._build_image_path_map()

        # Read and process the CSV files
        self._process_csv_files(csv_files)

        # If subset_indices is provided, filter the samples accordingly
        if subset_indices is not None:
            original_length = len(self.samples)
            self.samples = [self.samples[i] for i in subset_indices]
            if self.verbose:
                print(f"Subset_indices provided. Reduced samples from {original_length} to {len(self.samples)}.")

    def _build_image_path_map(self):
        if self.verbose:
            print("Building image path mapping...")

        for root in self.root_dirs:
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        full_path = os.path.join(dirpath, filename)
                        # Compute the relative path with respect to the root directory
                        rel_path = os.path.relpath(full_path, root)
                        rel_path = rel_path.replace(os.sep, '/')  # Normalize to use forward slashes
                        if rel_path not in self.image_path_map:
                            self.image_path_map[rel_path] = full_path
                        else:
                            if self.verbose:
                                print(f"Warning: Duplicate image path detected for {rel_path}. Using the first occurrence.")

        if self.verbose:
            print(f"Total unique images found: {len(self.image_path_map)}")

    def _process_csv_files(self, csv_files):
        if self.verbose:
            print("Processing CSV files...")

        # Read CSV files using pandas
        dfs = []
        for filename in csv_files:
            try:
                df = pd.read_csv(filename)
                dfs.append(df)
                if self.verbose:
                    print(f"Loaded {len(df)} rows from {filename}.")
            except Exception as e:
                if self.verbose:
                    print(f"Error reading {filename}: {e}")
                continue  # Skip files that can't be read

        data = pd.concat(dfs, ignore_index=True)

        if self.verbose:
            print(f"Total rows after concatenation: {len(data)}")

        # Normalize the 'Path' column to have consistent relative paths
        data['Path'] = data['Path'].apply(self._normalize_path)

        # Extract patient_id and study_id from the normalized 'Path'
        data['patient_id'] = data['Path'].str.split('/').str[0]
        data['study_id'] = data['Path'].str.split('/').str[1]

        # Check if 'Frontal/Lateral' column exists
        if 'Frontal/Lateral' not in data.columns:
            if self.verbose:
                print("'Frontal/Lateral' column not found. Attempting to extract from 'Path'.")
            # Attempt to extract 'Frontal/Lateral' from 'Path'
            data['Frontal/Lateral'] = data['Path'].apply(self._extract_view_from_path)

        # Proceed to process data
        self._process_data(data)

        if self.verbose:
            print(f"Total samples built: {len(self.samples)}")

    def _normalize_path(self, path):
        # Remove any leading directories up to 'patientXXXX/...'
        parts = path.replace('\\', '/').split('/')
        # Find the index where 'patient' occurs
        patient_index = next((i for i, part in enumerate(parts) if part.startswith('patient')), 0)
        normalized_path = '/'.join(parts[patient_index:])
        return normalized_path

    def _extract_view_from_path(self, path):
        filename = os.path.basename(path).lower()
        if 'frontal' in filename:
            return 'Frontal'
        elif 'lateral' in filename:
            return 'Lateral'
        else:
            return 'Unknown'

    def _process_data(self, data):
        # Group data by patient_id and study_id
        grouped = data.groupby(['patient_id', 'study_id'])

        # Iterate over each group to build samples
        for (patient_id, study_id), group in tqdm(grouped, disable=not self.verbose, desc="Building samples"):
            # Find frontal and lateral images
            frontal_rows = group[group['Frontal/Lateral'] == 'Frontal']
            lateral_rows = group[group['Frontal/Lateral'] == 'Lateral']

            if frontal_rows.empty or lateral_rows.empty:
                # Skip if either view is missing
                continue

            # For simplicity, take the first frontal and lateral images
            frontal_row = frontal_rows.iloc[0]
            lateral_row = lateral_rows.iloc[0]

            # Get image paths
            frontal_rel_path = frontal_row['Path']
            lateral_rel_path = lateral_row['Path']

            frontal_full_path = self.image_path_map.get(frontal_rel_path)
            lateral_full_path = self.image_path_map.get(lateral_rel_path)

            if frontal_full_path is None or lateral_full_path is None:
                # if self.verbose:
                #     print(f"Image not found: {frontal_rel_path} or {lateral_rel_path}")
                continue 

            # Collect labels (assuming labels are consistent across views)
            labels = []
            for label in self.LABELS:
                score = frontal_row.get(label, '')
                labels.append(float(score) if pd.notnull(score) else 0.0)
            labels_tensor = torch.tensor(labels, dtype=torch.float)

            # Add sample to the list
            self.samples.append((frontal_full_path, lateral_full_path, labels_tensor))

            # Debugging: Print first few samples
            if self.verbose and len(self.samples) <= 5:
                print(f"Sample {len(self.samples)}:")
                print(f"  Frontal Image: {frontal_full_path}")
                print(f"  Lateral Image: {lateral_full_path}")
                print(f"  Labels: {labels_tensor}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frontal_path, lateral_path, labels = self.samples[idx]
        # Load images
        frontal_image = Image.open(frontal_path).convert('L')  # Convert to grayscale
        lateral_image = Image.open(lateral_path).convert('L')

        # Apply transforms if provided
        if self.transform:
            frontal_image = self.transform(frontal_image)
            lateral_image = self.transform(lateral_image)

        return frontal_image, lateral_image, labels


class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image1, image2, labels = self.subset[index]
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, labels

    def __len__(self):
        return len(self.subset)


def create_data_loader(train_root_dirs, train_csv_file, config):
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

    train_dataset = CheXpertDataset(
        csv_files=train_csv_file,
        root_dirs=train_root_dirs,
        transform=None,
        verbose=True  # Set to True to enable detailed logging
    )

    # val_dataset = CheXpertDataset(
    #     csv_files=val_csv_file,
    #     root_dirs=val_root_dir,
    #     transform=val_transform,
    #     verbose=True  # Set to True to enable detailed logging
    # )
    total_size = len(train_dataset)
    train_size = int(config['train_ratio'] * total_size)
    valid_size = total_size - train_size  # Ensures that all samples are used

    train_subset, valid_subset = random_split(
        train_dataset, 
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(config['batch_size'])  # Ensures reproducibility
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


def create_test_data_loader(test_root_dirs, test_csv_file, config):
    test_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Since images are grayscale
    ])

    test_dataset = CheXpertDataset(
        csv_files=test_csv_file,
        root_dirs=test_root_dirs,
        transform=test_transform,
        verbose=True  # Set to True to enable detailed logging
    )

    print(f'Test_dataset: {len(test_dataset)}')

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory']
    )

    return test_loader

def create_train_val_test_data_loader(train_root_dirs, train_csv_file, config):
    train_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Uncomment if normalization is needed
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Uncomment if normalization is needed
    ])

    full_dataset = CheXpertDataset(
        csv_files=train_csv_file,
        root_dirs=train_root_dirs,
        transform=None,
        verbose=True  # Enable detailed logging
    )

    total_size = len(full_dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.125 * total_size)
    test_size = total_size - train_size - val_size  # Ensures all samples are used

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['seed'])  # Ensures reproducibility
    )

    train_dataset = SubsetWithTransform(train_subset, transform=train_transform)
    val_dataset = SubsetWithTransform(val_subset, transform=val_test_transform)
    test_dataset = SubsetWithTransform(test_subset, transform=val_test_transform)

    # Print the sizes of each subset
    print(f'Train_dataset: {len(train_dataset)} | Val_dataset: {len(val_dataset)} | Test_dataset: {len(test_dataset)}')

    # Create DataLoader for the training dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    # Create DataLoader for the validation dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    # Create DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    # Return all three DataLoaders
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    seed = 42
    torch.manual_seed(seed)

    test_root_dir = ['/data/DERI-USMSK/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/valid',
                     '/data/DERI-USMSK/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/test']

    test_csv_file = ['/data/DERI-USMSK/chexpertchestxrays-u20210408/test_1.csv']

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Since images are grayscale
    ])

    test_dataset = CheXpertDataset(
        csv_files=test_csv_file,
        root_dirs=test_root_dir,
        transform=test_transform,
        verbose=True  # Set to True to enable detailed logging
    )

    print(f'Test_dataset: {len(test_dataset)}')

    train_ratio = 0.85
    valid_ratio = 0.15

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Since images are grayscale
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Since images are grayscale
    ])

    root_dirs =[
        '/data/DERI-USMSK/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)',
        '/data/DERI-USMSK/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 3 (train 2)',
        '/data/DERI-USMSK/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 4 (train 3)'
    ]
    # root_dirs =[
    #     '/data/DERI-USMSK/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)'  
    # ]
    val_root_dir = ['/data/DERI-USMSK/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/valid']

    train_csv_file = ['/data/DERI-USMSK/chexpertchestxrays-u20210408/train_1.csv']
    val_csv_file = ['/data/DERI-USMSK/chexpertchestxrays-u20210408/valid_1.csv']

    train_dataset = CheXpertDataset(
        csv_files=train_csv_file,
        root_dirs=root_dirs,
        transform=None,
        verbose=True  # Set to True to enable detailed logging
    )

    total_size = len(train_dataset)
    train_size = int(train_ratio * total_size)
    valid_size = total_size - train_size  # Ensures that all samples are used

    train_subset, valid_subset = random_split(
        train_dataset, 
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed)  # Ensures reproducibility
    )
    train_dataset = SubsetWithTransform(train_subset, transform=train_transform)
    valid_dataset = SubsetWithTransform(valid_subset, transform=val_transform)
    # val_dataset = CheXpertDataset(
    #     csv_files=val_csv_file,
    #     root_dirs=val_root_dir,
    #     transform=val_transform,
    #     verbose=True  # Set to True to enable detailed logging
    # )

    print(f'Train_dataset: {len(train_dataset)} Val_dataset: {len(valid_dataset)}')

