import os
import torch
import glob
import csv
import pydicom
import numpy as np
from PIL import Image
import skimage.transform
import skimage.measure
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import libs.nyu_cropping as nyu_cropping

ASSESSMENT_MAP = {
    'BENIGN_WITHOUT_CALLBACK': 'benign',
    'BENIGN': 'benign',
    'MALIGNANT': 'malignant'
}
ASSESSMENTS = ('benign', 'malignant')

SIDE_MAP = {
    'LEFT': 'left',
    'RIGHT': 'right'
}

VIEW_MAP = {
    'CC': 'cc',
    'MLO': 'mlo'
}

def get_class_weights(dataset, class_key='assessment_label', config=None):  
    classes = ['benign', 'malignant']
    y = [subj[class_key] for subj in dataset.subjects]
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(config['device'])
    return class_weights

def extract_series_uid(file_path):
    parts = file_path.strip().split('/')
    if len(parts) < 2:
        raise ValueError(f"Path '{file_path}' not in expected format.")
    return parts[-2]  # second-last part is presumably the Series UID


class DDSMXray_Dataset(Dataset):
    def __init__(
        self,
        main_csv_files,
        metadata_csv,
        base_dir,
        crop_size=None,
        rescale_factor=None,
        verbose=False
    ):
        super().__init__()
        self.main_csv_files = main_csv_files if isinstance(main_csv_files, list) else [main_csv_files]
        self.metadata_csv = metadata_csv
        self.base_dir = base_dir
        self.crop_size = crop_size
        self.rescale_factor = rescale_factor
        self.verbose = verbose

        # Map from Series UID -> file location from metadata
        self.series_uid_to_file_location = {}
        # Final list of subjects
        self.subjects = []

        self._load_metadata()
        self._load_and_group_main_csvs()

    def _print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _load_metadata(self):
        """Load your metadata CSV into a dict: {Series UID: File Location}."""
        self._print_verbose(f"Loading metadata from {self.metadata_csv}")
        with open(self.metadata_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Loading metadata.csv"):
                series_uid = row['Series UID']
                file_location = row['File Location']
                num_imgs = int(row['Number of Images'])
                self.series_uid_to_file_location[series_uid] = [file_location,num_imgs]
        self._print_verbose(f"Total Series UIDs loaded: {len(self.series_uid_to_file_location)}")

    def _find_dicom_in_dir(self, base_dir, img_num):
        search_path = os.path.join(base_dir, f'1-{img_num}.dcm')
        
        return search_path  
    
    def _find_crop_dicom_in_dir(self, base_dir, img_num):
        search_path = os.path.join(base_dir, f'1-{img_num}.dcm')
        
        return search_path 
    
    def _find_mask_dicom_in_dir(self, base_dir, img_num):
        search_path = os.path.join(base_dir, f'1-{img_num}.dcm')
        
        return search_path  

    def _normalize_csv_subdir(self, csv_path_str):
        path_unix = csv_path_str.replace('\\', '/')

        if path_unix.startswith('./'):
            path_unix = path_unix[2:]
        if path_unix.startswith('.//'):
            path_unix = path_unix[3:]

        directory_norm = os.path.normpath(path_unix) 

        return directory_norm
    
    def _load_and_group_main_csvs(self):
        subjects_dict = {}

        for csv_file in self.main_csv_files:
            self._print_verbose(f"Loading main CSV: {csv_file}")
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for line in tqdm(reader, desc=f"Processing {csv_file}"):

                    patient_id = line['patient_id']
                    side_str = line['left or right breast'].upper()
                    view_str = line['image view'].upper()
                    abnormality_type = line['abnormality type'].lower()
                    assessment_label = line['pathology']
                    
                    side = SIDE_MAP.get(side_str, side_str.lower())
                    view = VIEW_MAP.get(view_str, view_str.lower())

                    group_key = (patient_id, side, view, abnormality_type)

                    if group_key not in subjects_dict:
                        subjects_dict[group_key] = {
                            'patient_id': patient_id,
                            'side': side,
                            'view': view,
                            'density_score': 0,
                            'assessment_score': 0,
                            'assessment_label': assessment_label,
                            'abnormality_type': abnormality_type,
                            'image_file': None,
                            'crop_files': [],
                            'mask_files': []
                        }

                    # CSV paths
                    image_file_path_rel = line['image file path'].strip()
                    roi_mask_file_path_rel = line['ROI mask file path'].strip()
                    crop_file_path_rel = line['cropped image file path'].strip()

                    # Extract Series UID from the directory names
                    # (or skip if you do not rely on the UID approach)
                    image_series_uid = extract_series_uid(image_file_path_rel)
                    mask_series_uid = extract_series_uid(roi_mask_file_path_rel)
                    crop_series_uid = extract_series_uid(crop_file_path_rel)

                    # Lookup base directories in metadata
                    try:
                        image_dir_base, img_num = self.series_uid_to_file_location[image_series_uid]
                        mask_dir_base, mask_img_num = self.series_uid_to_file_location[mask_series_uid]
                        crop_dir_base, crop_img_num = self.series_uid_to_file_location[crop_series_uid]
                    except KeyError as e:
                        self._print_verbose(f"[WARNING] Series UID {e} not found in metadata. Skipping entry.")
                        continue

                    final_image_dir = os.path.join(self.base_dir, image_dir_base)
                    final_image_dir = self._normalize_csv_subdir(final_image_dir)
                    # Now search that directory for .dcm
                    real_image_dcm = self._find_dicom_in_dir(final_image_dir, img_num)
                    if not os.path.exists(real_image_dcm):
                        self._print_verbose(f"[WARNING] Image file not found in {final_image_dir}")
                        continue
                    
                    # For the crop image directory
                    final_crop_dir = os.path.join(self.base_dir, crop_dir_base)
                    final_crop_dir = self._normalize_csv_subdir(final_crop_dir)
                    real_crop_dcm = self._find_crop_dicom_in_dir(final_crop_dir, crop_img_num)
                    if not os.path.exists(real_crop_dcm):
                        self._print_verbose(f"[WARNING] Crop file not found in {final_crop_dir}")
                        continue
                    
                    # For the mask directory
                    final_mask_dir = os.path.join(self.base_dir, mask_dir_base)
                    final_mask_dir = self._normalize_csv_subdir(final_mask_dir)
                    real_mask_dcm = self._find_mask_dicom_in_dir(final_mask_dir, mask_img_num)
                    if not os.path.exists(real_mask_dcm):
                        self._print_verbose(f"[WARNING] Mask file not found in {final_mask_dir}")
                        continue

                    # Ensure have a single image_file
                    if subjects_dict[group_key]['image_file'] is None:
                        subjects_dict[group_key]['image_file'] = real_image_dcm
                    else:
                        if subjects_dict[group_key]['image_file'] != real_image_dcm:
                            self._print_verbose(
                                f"[WARNING] Another image file for {group_key}: {real_image_dcm} "
                                f"(already have {subjects_dict[group_key]['image_file']}). Keeping the first."
                            )

                    # Add this mask path
                    subjects_dict[group_key]['crop_files'].append(real_crop_dcm)
                    subjects_dict[group_key]['mask_files'].append(real_mask_dcm)


        # Now reorganize by (patient_id, side) to ensure both 'cc' and 'mlo'
        final_dict = {}
        for (pid, side, view, abnormality_type), data in subjects_dict.items():
            key2 = (pid, side, abnormality_type)
            if key2 not in final_dict:
                final_dict[key2] = {
                    'patient_id': pid,
                    'side': side,
                    'density_score': data['density_score'],
                    'assessment_score': data['assessment_score'],
                    'assessment_label': data['assessment_label'],
                    'abnormality_type': abnormality_type,
                    'views': {}
                }
            final_dict[key2]['views'][view] = {
                'image_file': data['image_file'],
                'mask_files': data['mask_files']
            }

        # Filter only those who have both 'cc' and 'mlo'
        for (pid, side, abnormality_type), val in final_dict.items():
            if 'cc' in val['views'] and 'mlo' in val['views']:
                cc_image_file = val['views']['cc']['image_file']
                mlo_image_file = val['views']['mlo']['image_file']
                if cc_image_file and mlo_image_file:
                    self.subjects.append({
                        'patient_id': pid,
                        'side': side,
                        'density_score': val['density_score'],
                        'assessment_score': val['assessment_score'],
                        'assessment_label': val['assessment_label'],
                        'abnormality_type': val['abnormality_type'],
                        'views': val['views']
                    })
                else:
                    self._print_verbose(f"[WARNING] Skipping {pid} {side} {abnormality_type} due to missing image files.")
            else:
                self._print_verbose(f"[INFO] Skipping {pid} {side} {abnormality_type} due to missing CC or MLO.")

        self._print_verbose(f"Total subjects with both CC and MLO: {len(self.subjects)}")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        cc_data = subj['views']['cc']
        mlo_data = subj['views']['mlo']
        side = subj['side']
        patient_id = subj['patient_id']
        
        cc_image, cc_mask = self._load_and_preprocess(cc_data, view='cc', side=side, patient_id=patient_id)
        mlo_image, mlo_mask = self._load_and_preprocess(mlo_data, view='mlo', side=side, patient_id=patient_id)
        if cc_image is None or mlo_image is None:
            raise RuntimeError(f"[ERROR] Could not process CC/MLO for subject {subj['patient_id']}")

        label_str = ASSESSMENT_MAP[subj['assessment_label']]
        if label_str == 'benign':
            label = 0 
        elif label_str == 'malignant':
            label = 1
        else:
            raise ValueError(f"Unexpected label: {label_str}")

        metadata = {
            'patient_id': subj['patient_id'],
            'side': subj['side'],
            'density_score': subj['density_score'],
            'assessment_score': subj['assessment_score'],
            'assessment_label': label_str,
            'abnormality_type': subj['abnormality_type']
        }
        return cc_image, mlo_image, label

    def _load_and_preprocess(self, view_data, view, side, patient_id):
        image_file = view_data.get('image_file')
        index_info = f"side: {side}, view: {view}, patient_id: {patient_id}"
        
        if image_file is None:
            error_msg = f"[ERROR] 'image_file' is None for {index_info}"
            print(error_msg)
            raise ValueError(error_msg)
        
        if not os.path.exists(image_file):
            error_msg = f"[ERROR] DICOM file does not exist at path: {image_file} for {index_info}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        ds = pydicom.dcmread(view_data['image_file'])
        image = ds.pixel_array.astype(np.float32)
        # scale to [-1, +1]
        # image = 2.0 * (image / 65535.0) - 1.0

        # Combine masks
        combined_mask = np.zeros_like(image, dtype=bool)
        for mask_path in view_data['mask_files']:
            ds_mask = pydicom.dcmread(mask_path)
            mask_img = ds_mask.pixel_array
            if mask_img.shape != image.shape:
                mask_img = skimage.transform.resize(mask_img, image.shape, order=0, preserve_range=True)
            combined_mask |= (mask_img > 0)

        # optional rescale
        if self.rescale_factor is not None:
            scale = self.rescale_factor
            image = skimage.transform.rescale(image, scale, order=3, preserve_range=True).astype(np.float32)
            combined_mask = skimage.transform.rescale(
                combined_mask.astype(np.float32),
                scale,
                order=0,
                preserve_range=True
            ).astype(bool)
        
        # optional crop
        if self.crop_size is not None:
            image, combined_mask = self._crop2(image, combined_mask, side=side, view=view)

        # image_tensor = torch.from_numpy(image).unsqueeze(0)
        # mask_tensor = torch.from_numpy(combined_mask.astype(np.uint8))
        # return image_tensor, mask_tensor
        return image, combined_mask


    def _crop2(self, image, mask, side, view):
        cropping_info = nyu_cropping.crop_img_from_largest_connected(image, side)
        top, bottom, left, right = cropping_info[0]
        cropped_image = image[top:bottom, left:right]
        cropped_mask = mask
        return cropped_image, cropped_mask


# Combine the image and mask ROI
class DDSMXray_Crop_Test_Dataset(Dataset):
    def __init__(
        self,
        main_csv_files,
        metadata_csv,
        base_dir,
        crop_size=None,
        rescale_factor=None,
        verbose=False
    ):
        super().__init__()
        self.main_csv_files = main_csv_files if isinstance(main_csv_files, list) else [main_csv_files]
        self.metadata_csv = metadata_csv
        self.base_dir = base_dir
        self.crop_size = crop_size
        self.rescale_factor = rescale_factor
        self.verbose = verbose
        self.num_skipped_missing_views = 0
        # Map from Series UID -> file location from metadata
        self.series_uid_to_file_location = {}
        # Final list of subjects
        self.subjects = []

        self._load_metadata()
        self._load_and_group_main_csvs()

    def _print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _load_metadata(self):
        """Load your metadata CSV into a dict: {Series UID: File Location}."""
        self._print_verbose(f"Loading metadata from {self.metadata_csv}")
        with open(self.metadata_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Loading metadata.csv"):
                series_uid = row['Series UID']
                file_location = row['File Location']
                num_imgs = int(row['Number of Images'])
                self.series_uid_to_file_location[series_uid] = [file_location, num_imgs]
        self._print_verbose(f"Total Series UIDs loaded: {len(self.series_uid_to_file_location)}")

    def _find_dicom_in_dir(self, base_dir, img_num):
        search_path = os.path.join(base_dir, f'1-{img_num}.dcm')
        return search_path  
    
    def _find_crop_dicom_in_dir(self, base_dir, img_num):
        search_path = os.path.join(base_dir, f'1-{img_num}.dcm')
        return search_path 
    
    def _find_mask_dicom_in_dir(self, base_dir, img_num):
        search_path = os.path.join(base_dir, f'1-{img_num}.dcm')
        return search_path  

    def _normalize_csv_subdir(self, csv_path_str):
        path_unix = csv_path_str.replace('\\', '/')
        if path_unix.startswith('./'):
            path_unix = path_unix[2:]
        if path_unix.startswith('.//'):
            path_unix = path_unix[3:]
        directory_norm = os.path.normpath(path_unix) 
        return directory_norm

    def get_num_unique_patients(self):
        unique_patient_ids = set(subject['patient_id'] for subject in self.subjects)
        return len(unique_patient_ids)

    def _load_and_group_main_csvs(self):
        subjects_dict = {}

        for csv_file in self.main_csv_files:
            self._print_verbose(f"Loading main CSV: {csv_file}")
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for line in tqdm(reader, desc=f"Processing {csv_file}"):
                    patient_id = line['patient_id']
                    side_str = line['left or right breast'].upper()
                    view_str = line['image view'].upper()
                    abnormality_type = line['abnormality type'].lower()
                    assessment_label = line['pathology']
                    
                    side = SIDE_MAP.get(side_str, side_str.lower())
                    view = VIEW_MAP.get(view_str, view_str.lower())

                    group_key = (patient_id, side, view, abnormality_type)

                    if group_key not in subjects_dict:
                        subjects_dict[group_key] = {
                            'patient_id': patient_id,
                            'side': side,
                            'view': view,
                            'density_score': 0,
                            'assessment_score': 0,
                            'assessment_label': assessment_label,
                            'abnormality_type': abnormality_type,
                            'image_file': None,
                            'crop_files': [],
                            'mask_files': []
                        }

                    image_file_path_rel = line['image file path'].strip()
                    roi_mask_file_path_rel = line['ROI mask file path'].strip()
                    crop_file_path_rel = line['cropped image file path'].strip()

                    image_series_uid = extract_series_uid(image_file_path_rel)
                    mask_series_uid = extract_series_uid(roi_mask_file_path_rel)
                    crop_series_uid = extract_series_uid(crop_file_path_rel)

                    try:
                        image_dir_base, img_num = self.series_uid_to_file_location[image_series_uid]
                        mask_dir_base, mask_img_num = self.series_uid_to_file_location[mask_series_uid]
                        crop_dir_base, crop_img_num = self.series_uid_to_file_location[crop_series_uid]
                    except KeyError as e:
                        self._print_verbose(f"[WARNING] Series UID {e} not found in metadata. Skipping entry.")
                        continue

                    final_image_dir = os.path.join(self.base_dir, image_dir_base)
                    final_image_dir = self._normalize_csv_subdir(final_image_dir)

                    real_image_dcm = self._find_dicom_in_dir(final_image_dir, img_num)
                    if not os.path.exists(real_image_dcm):
                        self._print_verbose(f"[WARNING] Image file not found in {final_image_dir}")
                        continue
                    
                    final_crop_dir = os.path.join(self.base_dir, crop_dir_base)
                    final_crop_dir = self._normalize_csv_subdir(final_crop_dir)
                    real_crop_dcm = self._find_crop_dicom_in_dir(final_crop_dir, crop_img_num)
                    if not os.path.exists(real_crop_dcm):
                        self._print_verbose(f"[WARNING] Crop file not found in {final_crop_dir}")
                        continue
                    
                    final_mask_dir = os.path.join(self.base_dir, mask_dir_base)
                    final_mask_dir = self._normalize_csv_subdir(final_mask_dir)
                    real_mask_dcm = self._find_mask_dicom_in_dir(final_mask_dir, mask_img_num)
                    if not os.path.exists(real_mask_dcm):
                        self._print_verbose(f"[WARNING] Mask file not found in {final_mask_dir}")
                        continue

                    if subjects_dict[group_key]['image_file'] is None:
                        subjects_dict[group_key]['image_file'] = real_image_dcm
                    else:
                        if subjects_dict[group_key]['image_file'] != real_image_dcm:
                            self._print_verbose(
                                f"[WARNING] Another image file for {group_key}: {real_image_dcm} "
                                f"(already have {subjects_dict[group_key]['image_file']}). Keeping the first."
                            )

                    subjects_dict[group_key]['crop_files'].append(real_crop_dcm)
                    subjects_dict[group_key]['mask_files'].append(real_mask_dcm)

        final_dict = {}
        for (pid, side, view, abnormality_type), data in subjects_dict.items():
            key2 = (pid, side, abnormality_type)
            if key2 not in final_dict:
                final_dict[key2] = {
                    'patient_id': pid,
                    'side': side,
                    'density_score': data['density_score'],
                    'assessment_score': data['assessment_score'],
                    'assessment_label': data['assessment_label'],
                    'abnormality_type': abnormality_type,
                    'views': {}
                }
            final_dict[key2]['views'][view] = {
                'image_file': data['image_file'],
                'mask_files': data['mask_files']
            }

        for (pid, side, abnormality_type), val in final_dict.items():
            if 'cc' in val['views'] and 'mlo' in val['views']:
                cc_image_file = val['views']['cc']['image_file']
                mlo_image_file = val['views']['mlo']['image_file']
                if cc_image_file and mlo_image_file:
                    self.subjects.append({
                        'patient_id': pid,
                        'side': side,
                        'density_score': val['density_score'],
                        'assessment_score': val['assessment_score'],
                        'assessment_label': val['assessment_label'],
                        'abnormality_type': val['abnormality_type'],
                        'views': val['views']
                    })
                else:
                    self._print_verbose(f"[WARNING] Skipping {pid} {side} {abnormality_type} due to missing image files.")
                    self.num_skipped_missing_views += 1
            else:
                self._print_verbose(f"[INFO] Skipping {pid} {side} {abnormality_type} due to missing CC or MLO.")
                self.num_skipped_missing_views += 1

        self._print_verbose(f"Total subjects with both CC and MLO: {len(self.subjects)}")
        self._print_verbose(f"Total skipped due to missing CC or MLO: {self.num_skipped_missing_views}")
        
    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        cc_data = subj['views']['cc']
        mlo_data = subj['views']['mlo']
        side = subj['side']
        patient_id = subj['patient_id']
        
        cc_image, cc_mask, cc_cropping_info = self._load_and_preprocess(cc_data, view='cc', side=side, patient_id=patient_id)
        mlo_image, mlo_mask, mlo_cropping_info = self._load_and_preprocess(mlo_data, view='mlo', side=side, patient_id=patient_id)
        if cc_image is None or mlo_image is None:
            raise RuntimeError(f"[ERROR] Could not process CC/MLO for subject {subj['patient_id']}")

        label_str = ASSESSMENT_MAP[subj['assessment_label']]
        if label_str == 'benign':
            label = 0 
        elif label_str == 'malignant':
            label = 1
        else:
            raise ValueError(f"Unexpected label: {label_str}")

        metadata = {
            'patient_id': subj['patient_id'],
            'side': subj['side'],
            'density_score': subj['density_score'],
            'assessment_score': subj['assessment_score'],
            'assessment_label': label_str,
            'abnormality_type': subj['abnormality_type']
        }
        return cc_image, mlo_image, label

    def _load_and_preprocess(self, view_data, view, side, patient_id):
        image_file = view_data.get('image_file')
        index_info = f"side: {side}, view: {view}, patient_id: {patient_id}"
        
        if image_file is None:
            error_msg = f"[ERROR] 'image_file' is None for {index_info}"
            print(error_msg)
            raise ValueError(error_msg)
        
        if not os.path.exists(image_file):
            error_msg = f"[ERROR] DICOM file does not exist at path: {image_file} for {index_info}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        ds = pydicom.dcmread(image_file)
        image = ds.pixel_array.astype(np.float32)

        combined_mask = np.zeros_like(image, dtype=bool)
        for mask_path in view_data['mask_files']:
            ds_mask = pydicom.dcmread(mask_path)
            mask_img = ds_mask.pixel_array
            if mask_img.shape != image.shape:
                mask_img = skimage.transform.resize(mask_img, image.shape, order=0, preserve_range=True)
            combined_mask |= (mask_img > 0)

        if self.rescale_factor is not None:
            scale = self.rescale_factor
            image = skimage.transform.rescale(image, scale, order=3, preserve_range=True).astype(np.float32)
            combined_mask = skimage.transform.rescale(
                combined_mask.astype(np.float32),
                scale,
                order=0,
                preserve_range=True
            ).astype(bool)
            
        if self.crop_size is not None:
            image, combined_mask, cropping_info = self._crop2(image, combined_mask, side=side, view=view)
        else:
            cropping_info = None

        overlayed_image = self._overlay_mask(image, combined_mask, color=[255, 255, 255], alpha=0.7)
        
        return overlayed_image, combined_mask, cropping_info

    def _crop2(self, image, mask, side, view):
        cropping_info = nyu_cropping.crop_img_from_largest_connected(image, side)
        top, bottom, left, right = cropping_info[0]
        cropped_image = image[top:bottom, left:right]
        cropped_mask = mask[top:bottom, left:right]  # Adjust if you need to crop the mask similarly.
        return cropped_image, cropped_mask, cropping_info

    def _overlay_mask(self, image, mask, color=[255, 255, 255], alpha=0.5):
        """
        Create an overlay image where the ROI (as defined by mask) is highlighted.
        
        Parameters:
            image (ndarray): Input grayscale image (assumed to be 2D).
            mask (ndarray): Boolean mask with the same spatial dimensions as image.
            color (list or tuple): The RGB color to use for highlighting the ROI.
            alpha (float): The blending factor for the overlay.
            
        Returns:
            combined (ndarray): The RGB image with the ROI overlaid.
        """
    
        if image.ndim == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image.copy()
        
        if image_rgb.dtype != np.uint8:
            image_rgb = 255 * (image_rgb - np.min(image_rgb)) / (np.max(image_rgb) - np.min(image_rgb) + 1e-8)
            image_rgb = image_rgb.astype(np.uint8)
        
        overlay = image_rgb.copy()
        overlay[mask] = color
        
        combined = (alpha * overlay + (1 - alpha) * image_rgb).astype(np.uint8)
        return combined


class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image1_np, image2_np, labels = self.subset[index]
        
        if self.transform:
            image1_pil = Image.fromarray(image1_np)
            image2_pil = Image.fromarray(image2_np)
            image1_t = self.transform(image1_pil)
            image2_t = self.transform(image2_pil)
        # image1_pil = image1_pil.resize((224, 224))
        else:           
            image1_t = torch.from_numpy(image1_np).unsqueeze(0)
            image2_t = torch.from_numpy(image2_np).unsqueeze(0)
            
        return image1_t, image2_t, labels

    def __len__(self):
        return len(self.subset)

def create_ddsmxray_data_loader(train_val_main_csv_files, metadata_csv, base_dir, config, crop_size=None, rescale_factor=None, verbose=False):
    # Create train and val dataloaders
    train_transform = transforms.Compose([
        # transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Uncomment if normalization is needed
    ])

    val_test_transform = transforms.Compose([
        # transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Uncomment if normalization is needed
    ])
    
    ddsmxray_dataset = DDSMXray_Dataset(
        train_val_main_csv_files,
        metadata_csv,
        base_dir,
        crop_size=crop_size,
        rescale_factor=rescale_factor,
        verbose=verbose
    )
    
    print("Full dataset length:", len(ddsmxray_dataset))
    
    calc_indices = []
    mass_indices = []
    
    for i in range(len(ddsmxray_dataset)):
        abnormality = ddsmxray_dataset.subjects[i]['abnormality_type']
        
        if abnormality == "calcification":
            calc_indices.append(i)
        elif abnormality == "mass":
            mass_indices.append(i)
        else:
            # If there's some unexpected type, handle it or skip
            pass
    
    print(f"Number of calcification samples: {len(calc_indices)}")
    print(f"Number of mass samples:         {len(mass_indices)}")
    
    random.seed(42)
    random.shuffle(calc_indices)
    random.shuffle(mass_indices)
    
    calc_split = int(0.8 * len(calc_indices))  # 80% for train
    mass_split = int(0.8 * len(mass_indices))
    
    calc_train_indices = calc_indices[:calc_split]
    calc_val_indices   = calc_indices[calc_split:]
    mass_train_indices = mass_indices[:mass_split]
    mass_val_indices   = mass_indices[mass_split:]
    
    train_indices = calc_train_indices + mass_train_indices
    val_indices   = calc_val_indices   + mass_val_indices

    print(f"Train set size: {len(train_indices)}")
    print(f"Val   set size: {len(val_indices)}")

    train_dataset = Subset(ddsmxray_dataset, train_indices)
    val_dataset   = Subset(ddsmxray_dataset, val_indices)
    
    train_dataset = SubsetWithTransform(train_dataset, transform=train_transform)
    val_dataset = SubsetWithTransform(val_dataset, transform=val_test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=config['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory']
    )
    
    # train_class_weights = get_class_weights(train_dataset, class_key='assessment_label', config=config)
    
    return train_loader, val_loader


def create_ddsm_test_data_loader(test_main_csv_files, metadata_csv, base_dir, config, crop_size=None, rescale_factor=None, verbose=False):
    val_test_transform = transforms.Compose([
        # transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Uncomment if normalization is needed
    ])
    
    test_dataset = DDSMXray_Dataset(
        test_main_csv_files,
        metadata_csv,
        base_dir,
        crop_size=crop_size,
        rescale_factor=rescale_factor,
        verbose=verbose
    )

    print(f'Test_dataset: {len(test_dataset)}')

    test_dataset = SubsetWithTransform(test_dataset, transform=val_test_transform)
    
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory']
    )

    return test_loader

def create_ddsm_test_data_loader2(test_main_csv_files, metadata_csv, base_dir, config, crop_size=None, rescale_factor=None, verbose=False):
    val_test_transform = transforms.Compose([
        # transforms.Grayscale(1),
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Uncomment if normalization is needed
    ])
    
    test_dataset = DDSMXray_Crop_Test_Dataset(
        test_main_csv_files,
        metadata_csv,
        base_dir,
        crop_size=crop_size,
        rescale_factor=rescale_factor,
        verbose=verbose
    )

    print(f'Test_dataset: {len(test_dataset)}')

    test_dataset = SubsetWithTransform(test_dataset, transform=val_test_transform)
    
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory']
    )

    return test_loader


if __name__ == "__main__":
    main_csv_files = [
        "/data/DERI-USMSK/DDSMXray/calc_case_description_train_set.csv",
        "/data/DERI-USMSK/DDSMXray/mass_case_description_train_set.csv"
        #"/data/DERI-USMSK/DDSMXray/calc_case_description_test_set.csv",
        #"/data/DERI-USMSK/DDSMXray/mass_case_description_test_set.csv"
    ]
    metadata_csv = "/data/DERI-USMSK/DDSMXray/metadata.csv"
    base_dir = "/data/DERI-USMSK/DDSMXray/CBIS-DDSM"
    dataset = DDSMXray_Dataset(
        main_csv_files,
        metadata_csv,
        base_dir,
        crop_size= None,
        rescale_factor=None,
        verbose=True
    )
    print("Dataset length:", len(dataset))
    
    cc_image_tensor, mlo_image_tensor, label = dataset[1]

    # cc_image_tensor is shape (1, H, W) in torch
    # Convert to numpy for plotting
    cc_image_np = cc_image_tensor[0].numpy()  # shape (H, W)

    plt.figure()
    plt.imshow(cc_image_np, cmap='gray')
    plt.title(f"Label = {label} (0=benign, 1=malignant)")
    plt.show()
    
    create_ddsmxray_data_loader(main_csv_files, metadata_csv, base_dir, batch_size=16, crop_size=None, rescale_factor=None, verbose=True)
    
    
