import os
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import torch


class MURADataset(Dataset):
    def __init__(self, image_paths_csv, study_labels_csv, root_dir, body_parts=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # Each sample is (image_path1, image_path2, label)
        self.body_parts = body_parts

        # Read the image paths CSV file
        image_paths_df = pd.read_csv(image_paths_csv)

        # Read the study labels CSV file
        study_labels_df = pd.read_csv(study_labels_csv)
        study_labels_df['label'] = study_labels_df['label'].astype(int)

        # Filter based on the body parts if specified
        if self.body_parts:
            if isinstance(self.body_parts, str):
                body_parts_list = [self.body_parts]
            else:
                body_parts_list = self.body_parts

            # Create a regex pattern to match any of the specified body parts
            pattern = '|'.join(body_parts_list)

            # Filter image paths DataFrame
            image_paths_df = image_paths_df[image_paths_df['image_path'].str.contains(pattern)]

            # Filter study labels DataFrame
            study_labels_df = study_labels_df[study_labels_df['study_path'].str.contains(pattern)]

        # Create a mapping from study paths to labels
        study_to_label = {}
        for index, row in study_labels_df.iterrows():
            study_path = row['study_path'].rstrip('/').rstrip('\\')
            study_path = os.path.normpath(study_path)
            study_to_label[study_path] = row['label']

        # Build a mapping from study paths to image paths
        study_to_images = {}
        for index, row in image_paths_df.iterrows():
            image_path = os.path.join(self.root_dir, row['image_path'])
            image_path = os.path.normpath(image_path)

            # Extract the study path from the image path
            study_path = os.path.dirname(image_path)
            relative_study_path = os.path.relpath(study_path, self.root_dir)
            relative_study_path = relative_study_path.rstrip('/').rstrip('\\')
            relative_study_path = os.path.normpath(relative_study_path)

            # Get the label from the study_to_label mapping
            label = study_to_label.get(relative_study_path)
            if label is not None:
                if relative_study_path not in study_to_images:
                    study_to_images[relative_study_path] = []
                study_to_images[relative_study_path].append(image_path)
            else:
                print(f"Label not found for study: {relative_study_path}")

        # Generate pairs of images for each study
        for study_path, images in study_to_images.items():
            images.sort()
            label = study_to_label[study_path]
            N = len(images)

            if N == 1:
                # Duplicate the image
                img1 = images[0]
                img2 = images[0]
                self.samples.append((img1, img2, label))
            elif N == 2:
                # Pair the two images
                img1 = images[0]
                img2 = images[1]
                self.samples.append((img1, img2, label))
            elif N == 3:
                # Pair (image1, image2) and (image2, image3)
                img1 = images[0]
                img2 = images[1]
                self.samples.append((img1, img2, label))
                img1 = images[0]
                img2 = images[2]
                self.samples.append((img1, img2, label))
                img1 = images[1]
                img2 = images[2]
                self.samples.append((img1, img2, label))
            elif N >= 4:
                # Pair (image1, image2), (image1, image3), (image1, image4), (image2, image3), (image3, image4)
                img1 = images[0]
                # Pairs with image1
                for j in range(1, N):
                    img2 = images[j]
                    self.samples.append((img1, img2, label))
                # Pairs of consecutive images starting from image2
                for i in range(1, N):
                    for j in range(i + 1, N):
                        img1 = images[i]
                        img2 = images[j]
                        self.samples.append((img1, img2, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path1, image_path2, label = self.samples[idx]
        img1 = Image.open(image_path1).convert('L')  # Grayscale
        img2 = Image.open(image_path2).convert('L')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # create a tensor of shape (2, H, W)
        # img_pair = torch.stack([img1, img2], dim=0)
        return img1, img2, label

def create_data_loader(train_image_paths_csv, train_study_labels_csv, valid_image_paths_csv, valid_study_labels_csv, body_parts, config):
    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        # transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224])
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        # transforms.RandomCrop(224),
        # transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224])
    ])
    
    train_dataset = MURADataset(
        image_paths_csv=train_image_paths_csv,
        study_labels_csv=train_study_labels_csv,
        root_dir=config['root_dir'],
        body_parts=body_parts,
        transform=train_transform
    )

    valid_dataset = MURADataset(
        image_paths_csv=valid_image_paths_csv,
        study_labels_csv=valid_study_labels_csv,
        root_dir=config['root_dir'],
        body_parts=body_parts,
        transform=val_transform
    )   

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']
    )

    val_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']
    )

    print(f'Train_dataset: {len(train_dataset)} Val_dataset: {len(valid_dataset)}')

    return train_loader, val_loader

def create_test_data_loader(valid_image_paths_csv, valid_study_labels_csv, body_parts, config):
    test_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        # transforms.RandomCrop(224),
        # transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224])
    ])

    test_dataset = MURADataset(
        image_paths_csv=valid_image_paths_csv,
        study_labels_csv=valid_study_labels_csv,
        root_dir=config['root_dir'],
        body_parts=body_parts,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    return test_loader


class MURADataset2(Dataset):
    def __init__(self, image_paths_csv=None, study_labels_csv=None, root_dir=None, body_parts=None, transform=None, samples=None):
        """
        Args:
            image_paths_csv (str, optional): Path to the CSV file with image paths.
            study_labels_csv (str, optional): Path to the CSV file with study labels.
            root_dir (str): Root directory of the dataset.
            body_parts (list or str, optional): List of body parts to include.
            transform (callable, optional): Optional transform to be applied on a sample.
            samples (list of tuples, optional): Pre-split list of samples (image_path1, image_path2, label).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.body_parts = body_parts
        self.samples = samples if samples is not None else []

        if samples is None:
            # Initialize samples from CSVs
            if image_paths_csv is None or study_labels_csv is None:
                raise ValueError("If samples are not provided, image_paths_csv and study_labels_csv must be specified.")

            # Read the image paths CSV file
            image_paths_df = pd.read_csv(image_paths_csv)

            # Read the study labels CSV file
            study_labels_df = pd.read_csv(study_labels_csv)
            study_labels_df['label'] = study_labels_df['label'].astype(int)

            # Filter based on the body parts if specified
            if self.body_parts:
                if isinstance(self.body_parts, str):
                    body_parts_list = [self.body_parts]
                else:
                    body_parts_list = self.body_parts

                # Create a regex pattern to match any of the specified body parts
                pattern = '|'.join(body_parts_list)

                # Filter image paths DataFrame
                image_paths_df = image_paths_df[image_paths_df['image_path'].str.contains(pattern)]

                # Filter study labels DataFrame
                study_labels_df = study_labels_df[study_labels_df['study_path'].str.contains(pattern)]

            # Create a mapping from study paths to labels
            study_to_label = {}
            for index, row in study_labels_df.iterrows():
                study_path = row['study_path'].rstrip('/').rstrip('\\')
                study_path = os.path.normpath(study_path)
                study_to_label[study_path] = row['label']

            # Build a mapping from study paths to image paths
            study_to_images = {}
            for index, row in image_paths_df.iterrows():
                image_path = os.path.join(self.root_dir, row['image_path'])
                image_path = os.path.normpath(image_path)

                # Extract the study path from the image path
                study_path = os.path.dirname(image_path)
                relative_study_path = os.path.relpath(study_path, self.root_dir)
                relative_study_path = relative_study_path.rstrip('/').rstrip('\\')
                relative_study_path = os.path.normpath(relative_study_path)

                # Get the label from the study_to_label mapping
                label = study_to_label.get(relative_study_path)
                if label is not None:
                    if relative_study_path not in study_to_images:
                        study_to_images[relative_study_path] = []
                    study_to_images[relative_study_path].append(image_path)
                else:
                    print(f"Label not found for study: {relative_study_path}")

            # Generate pairs of images for each study
            for study_path, images in study_to_images.items():
                images.sort()
                label = study_to_label[study_path]
                N = len(images)

                if N == 1:
                    # Duplicate the image
                    img1 = images[0]
                    img2 = images[0]
                    self.samples.append((img1, img2, label))
                elif N == 2:
                    # Pair the two images
                    img1, img2 = images[0], images[1]
                    self.samples.append((img1, img2, label))
                elif N == 3:
                    # Pair (image1, image2) and (image2, image3)
                    img1 = images[0]
                    img2 = images[1]
                    self.samples.append((img1, img2, label))
                    img1 = images[0]
                    img2 = images[2]
                    self.samples.append((img1, img2, label))
                    img1 = images[1]
                    img2 = images[2]
                    self.samples.append((img1, img2, label))
                elif N >= 4:
                    # Pair (image1, image2), (image1, image3), (image1, image4), (image2, image3), (image3, image4)
                    img1 = images[0]
                    # Pairs with image1
                    for j in range(1, N):
                        img2 = images[j]
                        self.samples.append((img1, img2, label))
                    # Pairs of consecutive images starting from image2
                    for i in range(1, N):
                        for j in range(i + 1, N):
                            img1 = images[i]
                            img2 = images[j]
                            self.samples.append((img1, img2, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path1, image_path2, label = self.samples[idx]
        img1 = Image.open(image_path1).convert('L')  # Grayscale
        img2 = Image.open(image_path2).convert('L')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label
    

# def create_data_loader2(train_image_paths_csv, train_study_labels_csv, valid_image_paths_csv, valid_study_labels_csv, body_parts, config, save_csv=False, save_dir=None):
#     # Define transformations
#     train_transform = transforms.Compose([
#         transforms.Grayscale(1),
#         transforms.Resize(config['image_size']),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(30),
#         transforms.ToTensor(),
#         transforms.Normalize([0.456], [0.224])
#     ])

#     test_transform = transforms.Compose([
#         transforms.Grayscale(1),
#         transforms.Resize(config['image_size']),
#         transforms.ToTensor(),
#         transforms.Normalize([0.456], [0.224])
#     ])

#     val_transform = transforms.Compose([
#         transforms.Grayscale(1),
#         transforms.Resize(config['image_size']),
#         transforms.ToTensor(),
#         transforms.Normalize([0.456], [0.224])
#     ])
    
#     # ------------------- Prepare Training and Test Sets ------------------- #
#     # Read training CSVs
#     train_image_paths_df = pd.read_csv(train_image_paths_csv)
#     train_study_labels_df = pd.read_csv(train_study_labels_csv)
#     train_study_labels_df['label'] = train_study_labels_df['label'].astype(int)

#     # Filter based on the body parts if specified
#     if body_parts:
#         if isinstance(body_parts, str):
#             body_parts_list = [body_parts]
#         else:
#             body_parts_list = body_parts

#         # Create a regex pattern to match any of the specified body parts
#         pattern = '|'.join(body_parts_list)

#         # Filter image paths DataFrame
#         train_image_paths_df = train_image_paths_df[train_image_paths_df['image_path'].str.contains(pattern)]

#         # Filter study labels DataFrame
#         train_study_labels_df = train_study_labels_df[train_study_labels_df['study_path'].str.contains(pattern)]
#     else:
#         body_parts_list = []

#     # Create a mapping from study paths to labels
#     study_to_label = {}
#     for index, row in train_study_labels_df.iterrows():
#         study_path = row['study_path'].rstrip('/').rstrip('\\')
#         study_path = os.path.normpath(study_path)
#         study_to_label[study_path] = row['label']

#     # Build a mapping from study paths to image paths
#     study_to_images = {}
#     for index, row in train_image_paths_df.iterrows():
#         image_path = os.path.join(config['root_dir'], row['image_path'])
#         image_path = os.path.normpath(image_path)

#         # Extract the study path from the image path
#         study_path = os.path.dirname(image_path)
#         relative_study_path = os.path.relpath(study_path, config['root_dir'])
#         relative_study_path = relative_study_path.rstrip('/').rstrip('\\')
#         relative_study_path = os.path.normpath(relative_study_path)

#         # Get the label from the study_to_label mapping
#         label = study_to_label.get(relative_study_path)
#         if label is not None:
#             if relative_study_path not in study_to_images:
#                 study_to_images[relative_study_path] = []
#             study_to_images[relative_study_path].append(image_path)
#         else:
#             print(f"Label not found for study: {relative_study_path}")

#     # Generate samples as per the original Dataset class
#     samples = []
#     for study_path, images in study_to_images.items():
#         images.sort()
#         label = study_to_label[study_path]
#         N = len(images)

#         if N == 1:
#             # Duplicate the image
#             img1 = images[0]
#             img2 = images[0]
#             samples.append((img1, img2, label))
#         elif N == 2:
#             # Pair the two images
#             img1 = images[0]
#             img2 = images[1]
#             samples.append((img1, img2, label))
#         elif N == 3:
#             img1 = images[0]
#             img2 = images[1]
#             samples.append((img1, img2, label))
#             img1 = images[0]
#             img2 = images[2]
#             samples.append((img1, img2, label))
#             img1 = images[1]
#             img2 = images[2]
#             samples.append((img1, img2, label))
#         elif N >= 4:
#             img1 = images[0]
#             # Pairs with image1
#             for j in range(1, N):
#                 img2 = images[j]
#                 samples.append((img1, img2, label))
#             # Pairs of consecutive images starting from image2
#             for i in range(1, N):
#                 for j in range(i + 1, N):
#                     img1 = images[i]
#                     img2 = images[j]
#                     samples.append((img1, img2, label))

#     # Function to extract body part from image path
#     def get_body_part(image_path):
#         for part in body_parts_list:
#             if part in image_path:
#                 return part
#         return "UNKNOWN"

#     # Create a DataFrame from samples
#     sample_df = pd.DataFrame(samples, columns=['image_path1', 'image_path2', 'label'])

#     # Assign body part to each sample based on image_path1
#     sample_df['body_part'] = sample_df['image_path1'].apply(get_body_part)

#     # Initialize lists to hold train and test samples
#     train_samples = []
#     test_samples = []

#     # Perform stratified split: 90% train, 10% test per body part
#     for part, group in sample_df.groupby('body_part'):
#         if part == "UNKNOWN":
#             # Handle samples with unknown body parts if any
#             # Here, we choose to include all in training to avoid losing data
#             train_samples.append(group)
#             continue

#         # Determine the number of test samples (10% of the group)
#         test_size = max(1, int(0.08 * len(group)))  # Ensure at least one sample in test

#         # Split the group into train and test
#         part_train, part_test = train_test_split(
#             group,
#             test_size=test_size,
#             random_state=config['seed'],
#             shuffle=True
#         )

#         train_samples.append(part_train)
#         test_samples.append(part_test)

#     # Concatenate all train and test samples
#     train_samples = pd.concat(train_samples).reset_index(drop=True)
#     test_samples = pd.concat(test_samples).reset_index(drop=True)

#     if save_csv:
#         if save_dir is None:
#             save_dir = '.'  # Current directory
#         else:
#             os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

#         train_csv_path = os.path.join(save_dir, 'train_samples.csv')
#         test_csv_path = os.path.join(save_dir, 'test_samples.csv')

#         train_samples.to_csv(train_csv_path, index=False)
#         test_samples.to_csv(test_csv_path, index=False)

#         print(f"Train samples saved to '{train_csv_path}'")
#         print(f"Test samples saved to '{test_csv_path}'")

#     # Convert DataFrame rows to list of tuples
#     train_samples_list = list(train_samples[['image_path1', 'image_path2', 'label']].itertuples(index=False, name=None))
#     test_samples_list = list(test_samples[['image_path1', 'image_path2', 'label']].itertuples(index=False, name=None))

#     # ------------------- Initialize Datasets ------------------- #
#     # Training Dataset
#     train_dataset = MURADataset2(
#         samples=train_samples_list,
#         root_dir=config['root_dir'],
#         transform=train_transform
#     )

#     # Test Dataset
#     test_dataset = MURADataset2(
#         samples=test_samples_list,
#         root_dir=config['root_dir'],
#         transform=test_transform
#     )

#     # Validation Dataset (as per original code)
#     valid_dataset = MURADataset2(
#         image_paths_csv=valid_image_paths_csv,
#         study_labels_csv=valid_study_labels_csv,
#         root_dir=config['root_dir'],
#         body_parts=body_parts,
#         transform=val_transform
#     )

#     # ------------------- Create DataLoaders ------------------- #
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         num_workers=config['num_workers']
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=config['num_workers']
#     )

#     val_loader = DataLoader(
#         valid_dataset,
#         batch_size=config['batch_size'],
#         shuffle=False,
#         num_workers=config['num_workers']
#     )

#     print(f'Train_dataset: {len(train_dataset)} | Val_dataset: {len(valid_dataset)} | Test_dataset: {len(test_dataset)}')

#     return train_loader, val_loader, test_loader


def create_data_loader3(train_image_paths_csv, train_study_labels_csv, valid_image_paths_csv, valid_study_labels_csv, body_parts, config, save_csv=False, save_dir=None):
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224])
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224])
    ])
    
    # ------------------- Prepare Training and Test Sets ------------------- #
    # Read training CSVs
    train_image_paths_df = pd.read_csv(train_image_paths_csv)
    train_study_labels_df = pd.read_csv(train_study_labels_csv)
    train_study_labels_df['label'] = train_study_labels_df['label'].astype(int)

    # Filter based on the body parts if specified
    if body_parts:
        if isinstance(body_parts, str):
            body_parts_list = [body_parts]
        else:
            body_parts_list = body_parts

        # Create a regex pattern to match any of the specified body parts
        pattern = '|'.join(body_parts_list)

        # Filter image paths DataFrame
        train_image_paths_df = train_image_paths_df[train_image_paths_df['image_path'].str.contains(pattern, regex=True, na=False)]

        # Filter study labels DataFrame
        train_study_labels_df = train_study_labels_df[train_study_labels_df['study_path'].str.contains(pattern, regex=True, na=False)]
    else:
        body_parts_list = []

    # Create a mapping from study paths to labels
    study_to_label = {}
    for index, row in train_study_labels_df.iterrows():
        study_path = row['study_path'].rstrip('/').rstrip('\\')
        study_path = os.path.normpath(study_path)
        study_to_label[study_path] = row['label']

    # Build a mapping from study paths to image paths
    study_to_images = {}
    for index, row in train_image_paths_df.iterrows():
        image_path = os.path.join(config['root_dir'], row['image_path'])
        image_path = os.path.normpath(image_path)

        # Extract the study path from the image path
        study_path = os.path.dirname(image_path)
        relative_study_path = os.path.relpath(study_path, config['root_dir'])
        relative_study_path = relative_study_path.rstrip('/').rstrip('\\')
        relative_study_path = os.path.normpath(relative_study_path)

        # Get the label from the study_to_label mapping
        label = study_to_label.get(relative_study_path)
        if label is not None:
            if relative_study_path not in study_to_images:
                study_to_images[relative_study_path] = []
            study_to_images[relative_study_path].append(image_path)
        else:
            print(f"Label not found for study: {relative_study_path}")

    # Generate samples as per the original Dataset class
    samples = []
    for study_path, images in study_to_images.items():
        images.sort()
        label = study_to_label[study_path]
        N = len(images)

        if N == 1:
            # Duplicate the image
            img1 = images[0]
            img2 = images[0]
            samples.append((img1, img2, label, study_path))
        elif N == 2:
            # Pair the two images
            img1 = images[0]
            img2 = images[1]
            samples.append((img1, img2, label, study_path))
        elif N == 3:
            samples.append((images[0], images[1], label, study_path))
            samples.append((images[0], images[2], label, study_path))
            samples.append((images[1], images[2], label, study_path))
        elif N >= 4:
            img1 = images[0]
            # Pairs with image1
            for j in range(1, N):
                img2 = images[j]
                samples.append((img1, img2, label, study_path))
            # Pairs of consecutive images starting from image2
            for i in range(1, N):
                for j in range(i + 1, N):
                    img1 = images[i]
                    img2 = images[j]
                    samples.append((img1, img2, label, study_path))

    # Function to extract body part from image path
    def get_body_part(image_path):
        for part in body_parts_list:
            if part in image_path:
                return part
        return "UNKNOWN"

    # Create a DataFrame from samples
    sample_df = pd.DataFrame(samples, columns=['image_path1', 'image_path2', 'label', 'study_path'])

    # Assign body part to each sample based on image_path1
    sample_df['body_part'] = sample_df['image_path1'].apply(get_body_part)

    # ------------------- Split Studies into Train and Test ------------------- #
    # Extract unique studies with their body parts
    study_df = pd.DataFrame({
        'study_path': list(study_to_images.keys()),
        'label': [study_to_label[sp] for sp in study_to_images.keys()]
    })

    # If body parts are specified, assign body parts to studies
    if body_parts:
        def extract_body_part_from_study(study_path):
            for part in body_parts_list:
                if part in study_path:
                    return part
            return "UNKNOWN"

        study_df['body_part'] = study_df['study_path'].apply(extract_body_part_from_study)
    else:
        study_df['body_part'] = "ALL"

    # Perform stratified split based on body parts
    if body_parts:
        stratify_col = study_df['body_part']
    else:
        stratify_col = None

    train_studies, test_studies = train_test_split(
        study_df,
        test_size=0.08,  # 10% for testing
        random_state=config['seed'],
        shuffle=True,
        stratify=stratify_col
    )

    # Get list of study paths for train and test
    train_study_paths = set(train_studies['study_path'])
    test_study_paths = set(test_studies['study_path'])

    # Assign samples to train and test based on study assignment
    train_samples = sample_df[sample_df['study_path'].isin(train_study_paths)].copy()
    test_samples = sample_df[sample_df['study_path'].isin(test_study_paths)].copy()

    # Handle "UNKNOWN" body parts: Assign them entirely to training to avoid leakage
    unknown_samples = sample_df[sample_df['body_part'] == "UNKNOWN"]
    train_samples = pd.concat([train_samples, unknown_samples], ignore_index=True)

    # Optionally save to CSV
    if save_csv:
        if save_dir is None:
            save_dir = '.'  # Current directory
        else:
            os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

        train_csv_path = os.path.join(save_dir, 'train_samples.csv')
        test_csv_path = os.path.join(save_dir, 'test_samples.csv')

        train_samples.to_csv(train_csv_path, index=False)
        test_samples.to_csv(test_csv_path, index=False)

        print(f"Train samples saved to '{train_csv_path}'")
        print(f"Test samples saved to '{test_csv_path}'")

    # Convert DataFrame rows to list of tuples
    train_samples_list = list(train_samples[['image_path1', 'image_path2', 'label']].itertuples(index=False, name=None))
    test_samples_list = list(test_samples[['image_path1', 'image_path2', 'label']].itertuples(index=False, name=None))
    overlap = set(train_samples_list) & set(test_samples_list)
    print(f"Overlap: {len(overlap)}")
    # ------------------- Initialize Datasets ------------------- #
    # Training Dataset
    train_dataset = MURADataset2(
        samples=train_samples_list,
        root_dir=config['root_dir'],
        transform=train_transform
    )

    # Test Dataset
    test_dataset = MURADataset2(
        samples=test_samples_list,
        root_dir=config['root_dir'],
        transform=test_transform
    )

    # Validation Dataset (as per original code)
    valid_dataset = MURADataset2(
        image_paths_csv=valid_image_paths_csv,
        study_labels_csv=valid_study_labels_csv,
        root_dir=config['root_dir'],
        body_parts=body_parts,
        transform=val_transform
    )

    # ------------------- Create DataLoaders ------------------- #
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers']
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    print(f'Train_dataset: {len(train_dataset)} | Val_dataset: {len(valid_dataset)} | Test_dataset: {len(test_dataset)}')

    return train_loader, val_loader, test_loader


def create_data_loader4(
    train_image_paths_csv,
    train_study_labels_csv,
    valid_image_paths_csv,
    valid_study_labels_csv,
    body_parts,
    config,
    save_csv=False,
    save_dir=None
):
    # ------------------- Define Transformations ------------------- #
    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224])
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224])
    ])
    
    # ------------------- Read and Process CSV Files ------------------- #
    # Read training image paths with header inference
    train_image_paths_df = pd.read_csv(
        train_image_paths_csv,
        header=0,  # Skip header
        names=['image_path'],
        skipinitialspace=True
    )
    print(f"Total image paths read: {len(train_image_paths_df)}")

    # Read training study labels with header inference
    train_study_labels_df = pd.read_csv(
        train_study_labels_csv,
        header=0,  # Skip header
        names=['study_path', 'label'],
        skipinitialspace=True
    )
    print(f"Total study labels read: {len(train_study_labels_df)}")

    # Convert 'label' to int, handling potential errors
    try:
        train_study_labels_df['label'] = train_study_labels_df['label'].astype(int)
    except ValueError as ve:
        print(f"ValueError during label conversion: {ve}")
        # Identify and remove problematic rows
        problematic_rows = train_study_labels_df[~train_study_labels_df['label'].apply(lambda x: str(x).isdigit())]
        print("Problematic Rows:")
        print(problematic_rows)
        # Remove problematic rows
        train_study_labels_df = train_study_labels_df[train_study_labels_df['label'].apply(lambda x: str(x).isdigit())]
        train_study_labels_df['label'] = train_study_labels_df['label'].astype(int)
        print(f"Total study labels after cleaning: {len(train_study_labels_df)}")

    # ------------------- Extract or Assign patient_id ------------------- #
    # Function to extract patient_id from study_path
    def extract_patient_id(study_path):
        parts = study_path.strip('/').split(os.sep)
        for part in parts:
            if part.lower().startswith("patient"):
                return part
        return "UNKNOWN_PATIENT"

    # Apply the function to extract patient_id
    # Make study_path absolute by joining with root_dir
    train_study_labels_df['study_path'] = train_study_labels_df['study_path'].apply(
        lambda x: os.path.normpath(os.path.join(config['root_dir'], x))
    )

    train_study_labels_df['patient_id'] = train_study_labels_df['study_path'].apply(extract_patient_id)


    # ------------------- Filter Based on Body Parts (If Specified) ------------------- #
    if body_parts:
        if isinstance(body_parts, str):
            body_parts_list = [body_parts]
        else:
            body_parts_list = body_parts

        # Create a regex pattern to match any of the specified body parts (case-insensitive)
        pattern = '|'.join([re.escape(part) for part in body_parts_list])

        # Filter image paths DataFrame
        train_image_paths_df = train_image_paths_df[
            train_image_paths_df['image_path'].str.contains(pattern, regex=True, na=False)
        ]
        # Filter study labels DataFrame
        train_study_labels_df = train_study_labels_df[
            train_study_labels_df['study_path'].str.contains(pattern, regex=True, na=False)
        ]
    else:
        body_parts_list = []

    # ------------------- Create Mappings ------------------- #
    # Mapping from study_path to label and patient_id
    study_to_label = dict(zip(train_study_labels_df['study_path'], train_study_labels_df['label']))
    study_to_patient = dict(zip(train_study_labels_df['study_path'], train_study_labels_df['patient_id']))

    # Mapping from study_path to list of image_paths
    study_to_images = {}
    unmatched_image_paths = 0
    for _, row in train_image_paths_df.iterrows():
        image_path = os.path.normpath(os.path.join(config['root_dir'], row['image_path']))
        study_path = os.path.normpath(os.path.dirname(image_path))

        # Get label and patient_id
        label = study_to_label.get(study_path)
        patient_id = study_to_patient.get(study_path, "UNKNOWN_PATIENT")

        if label is not None:
            if study_path not in study_to_images:
                study_to_images[study_path] = []
            study_to_images[study_path].append(image_path)
        else:
            unmatched_image_paths += 1
            print(f"Label not found for study: {study_path} (Image: {image_path})")

    # ------------------- Generate Samples Including patient_id ------------------- #
    samples = []
    for study_path, images in study_to_images.items():
        images.sort()
        label = study_to_label[study_path]
        patient_id = study_to_patient[study_path]
        N = len(images)

        if N == 1:
            # Duplicate the image
            img1 = images[0]
            img2 = images[0]
            samples.append((img1, img2, label, study_path, patient_id))
        elif N == 2:
            # Pair the two images
            img1, img2 = images
            samples.append((img1, img2, label, study_path, patient_id))
        elif N == 3:
            # All possible pairs
            samples.append((images[0], images[1], label, study_path, patient_id))
            samples.append((images[0], images[2], label, study_path, patient_id))
            samples.append((images[1], images[2], label, study_path, patient_id))
        else:
            # For N >= 4, generate all unique pairs
            for i in range(N):
                for j in range(i + 1, N):
                    img1, img2 = images[i], images[j]
                    samples.append((img1, img2, label, study_path, patient_id))

    print(f"Total samples generated: {len(samples)}")

    # ------------------- Assign Body Parts to Each Sample ------------------- #
    def get_body_part(image_path):
        for part in body_parts_list:
            if part.lower() in image_path.lower():
                return part
        return "UNKNOWN"

    sample_df = pd.DataFrame(samples, columns=['image_path1', 'image_path2', 'label', 'study_path', 'patient_id'])
    sample_df['body_part'] = sample_df['image_path1'].apply(get_body_part)

    # ------------------- Identify Patients with "UNKNOWN" Body Parts ------------------- #
    # These patients will be entirely assigned to the training set
    unknown_patients = set(sample_df[sample_df['body_part'] == "UNKNOWN"]['patient_id'].unique())

    # ------------------- Create Patient-Level DataFrame ------------------- #
    # Assign a single label per patient using majority label
    patient_df = train_study_labels_df.groupby('patient_id')['label'].agg(lambda x: x.value_counts().idxmax()).reset_index()

    # ------------------- Assign Patients Exclusively to Train or Test ------------------- #
    # Assign 'UNKNOWN_PATIENT' to training set exclusively
    if "UNKNOWN_PATIENT" in patient_df['patient_id'].unique():
        print("Assigning 'UNKNOWN_PATIENT's to the training set.")
        unknown_patient_ids = set(patient_df[patient_df['patient_id'] == "UNKNOWN_PATIENT"]['patient_id'])
        train_patient_ids = set(unknown_patient_ids)
        # Exclude 'UNKNOWN_PATIENT' from splitting
        remaining_patients_df = patient_df[~patient_df['patient_id'].isin(unknown_patient_ids)].copy()
    else:
        train_patient_ids = set()
        remaining_patients_df = patient_df.copy()

    # If patients have multiple labels, 'patient_df' now has a single label per patient based on majority
    if len(remaining_patients_df) > 0:
        if 'label' in remaining_patients_df.columns and remaining_patients_df['label'].nunique() > 1:
            stratify_col = remaining_patients_df['label']
        else:
            stratify_col = None

        # Perform stratified split
        train_patients_split_df, test_patients_split_df = train_test_split(
            remaining_patients_df,
            test_size=0.08,  # 8% for testing
            random_state=config['seed'],
            shuffle=True,
            stratify=stratify_col
        )

        # Update train and test patient IDs
        train_patient_ids.update(set(train_patients_split_df['patient_id']))
        test_patient_ids = set(test_patients_split_df['patient_id'])
    else:
        print("No remaining patients to split.")
        test_patient_ids = set()

    # ------------------- Diagnostic Check: Overlapping Patients ------------------- #
    overlapping_patients = train_patient_ids.intersection(test_patient_ids)
    print(f"Number of overlapping patients: {len(overlapping_patients)}")
    assert len(overlapping_patients) == 0, "There are overlapping patients between train and test sets!"

    # ------------------- Assign Samples to Train and Test Based on Patient Assignment ------------------- #
    train_samples = sample_df[sample_df['patient_id'].isin(train_patient_ids)].copy()
    test_samples = sample_df[sample_df['patient_id'].isin(test_patient_ids)].copy()

    print(f"Number of train samples: {len(train_samples)}")
    print(f"Number of test samples: {len(test_samples)}")

    # ------------------- Handle "UNKNOWN" Body Parts Correctly ------------------- #
    if len(unknown_patients) > 0:
        # Ensure "UNKNOWN" samples are only in the training set
        test_samples = test_samples[test_samples['body_part'] != "UNKNOWN"]
        print(f"Number of 'UNKNOWN' samples in test set after filtering: {len(test_samples[test_samples['body_part'] == 'UNKNOWN'])}")
    else:
        print("No 'UNKNOWN' samples to handle.")

    # ------------------- Verify No Overlaps Between Train and Test Samples ------------------- #
    # Convert to list of tuples for overlap check
    train_samples_list = list(train_samples[['image_path1', 'image_path2', 'label']].itertuples(index=False, name=None))
    test_samples_list = list(test_samples[['image_path1', 'image_path2', 'label']].itertuples(index=False, name=None))

    # Check for overlaps
    overlap = set(train_samples_list) & set(test_samples_list)
    print(f"Number of overlapping samples: {len(overlap)}")
    if len(overlap) > 0:
        print("Sample overlapping entries:")
        for sample in list(overlap)[:5]:
            print(sample)
    else:
        print("No overlapping samples detected.")

    # Assert no overlap
    assert len(overlap) == 0, "There are overlapping samples between train and test sets!"

    # ------------------- Optionally Save the Splits to CSV ------------------- #
    if save_csv:
        if save_dir is None:
            save_dir = '.'  # Current directory
        else:
            os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

        train_csv_path = os.path.join(save_dir, 'train_samples.csv')
        test_csv_path = os.path.join(save_dir, 'test_samples.csv')

        # Save only the relevant columns
        train_samples.to_csv(train_csv_path, index=False)
        test_samples.to_csv(test_csv_path, index=False)

        print(f"Train samples saved to '{train_csv_path}'")
        print(f"Test samples saved to '{test_csv_path}'")

    # ------------------- Initialize Datasets ------------------- #
    # Convert DataFrame rows to list of tuples
    train_samples_list = list(train_samples[['image_path1', 'image_path2', 'label']].itertuples(index=False, name=None))
    test_samples_list = list(test_samples[['image_path1', 'image_path2', 'label']].itertuples(index=False, name=None))

    # Verify that train_samples_list is not empty
    if len(train_samples_list) == 0:
        raise ValueError("Training dataset is empty after splitting. Please check your data and splitting logic.")
    
    if len(test_samples_list) == 0:
        raise ValueError("Test dataset is empty after splitting. Please check your data and splitting logic.")


    # Initialize Datasets
    train_dataset = MURADataset2(
        samples=train_samples_list,
        root_dir=config['root_dir'],
        transform=train_transform
    )

    valid_dataset = MURADataset2(
        samples=test_samples_list,
        root_dir=config['root_dir'],
        transform=test_transform
    )

    # Initialize Validation Dataset (Assuming validation is handled separately)
    test_dataset = MURADataset2(
        image_paths_csv=valid_image_paths_csv,
        study_labels_csv=valid_study_labels_csv,
        root_dir=config['root_dir'],
        body_parts=body_parts,
        transform=val_transform
    )

    # ------------------- Create DataLoaders ------------------- #
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers']
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    print(f'Train_dataset: {len(train_dataset)} | Val_dataset: {len(valid_dataset)} | Test_dataset: {len(test_dataset)}')

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Paths
    root_dir = r'/data/DERI-USMSK/'   # Parent directory

    # CSV files
    train_image_paths_csv = os.path.join(root_dir, 'MURA-v1.1', 'train_image_paths.csv')
    train_study_labels_csv = os.path.join(root_dir, 'MURA-v1.1', 'train_labeled_studies.csv')
    valid_image_paths_csv = os.path.join(root_dir, 'MURA-v1.1', 'valid_image_paths.csv')
    valid_study_labels_csv = os.path.join(root_dir, 'MURA-v1.1', 'valid_labeled_studies.csv')

    body_parts = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Since images are grayscale
    ])

    train_dataset = MURADataset(
        image_paths_csv=train_image_paths_csv,
        study_labels_csv=train_study_labels_csv,
        root_dir=root_dir,
        body_parts=body_parts,
        transform=transform
    )

    valid_dataset = MURADataset(
        image_paths_csv=valid_image_paths_csv,
        study_labels_csv=valid_study_labels_csv,
        root_dir=root_dir,
        body_parts=body_parts,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False
    )
    print(f'Train_dataset: {len(train_dataset)} Val_dataset: {len(valid_dataset)}')

    all_us_labels = []
    for images, labels in train_loader:
        all_us_labels.extend(labels.tolist())

    

