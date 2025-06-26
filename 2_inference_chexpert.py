import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import cv2
from libs.config import get_config
from libs.dataset_chexpert_twoview import create_test_data_loader
from libs.dataset_chexpert_twoview import create_train_val_test_data_loader as create_train_val_test_data_loader_chexpert
from pytorch_grad_cam import GradCAM as pytorch_gradcam

from net_fusionmamba import ModelWrapper
from net_fusionmamba import TwoViewXFMambaTop as twoviewxfmamba
from net_fusionmamba import SingleViewMamba as singleviewmamba
from net_fusionmamba import TwoViewLateJoinMamba as twoviewjoinmamba
from net_fusionmamba import TwoViewEarlyFusionMamba as twoviewearlyfusionmamba


print("\n\n\n############################################################")
print("STARTING INFERENCE 🔍")
print(f"Python: {sys.version}\nPytorch: {torch.__version__}\nPytorch on GPU: {torch.cuda.is_available()}")
print("############################################################\n\n\n")

"""🔧 Parse command line arguments 🔧"""
parser = argparse.ArgumentParser(description='Inference for MSK image classification model')
parser.add_argument('--root_dir', type=str, default='.', help='Path to data directory')
parser.add_argument('--train_image_paths', type=str, default='train_image_paths.csv', help='Path to train image')
parser.add_argument('--train_image_labels', type=str, default='train_labeled_studies.csv', help='Path to train image labels')
parser.add_argument('--valid_image_paths', type=str, default='valid_image_paths.csv', help='Path to validation image')
parser.add_argument('--valid_image_labels', type=str, default='valid_labeled_studies.csv', help='Path to validation image labels')
parser.add_argument('--confusion_matrix_folder', type=str, default='output_confusionmatrix_mura_allbodyparts_fusionview_efffusionmambav4_2')
parser.add_argument('--dataset', type=str, default='chexpert', help='dataset selection')
parser.add_argument("--epochs", type=int, default=100, required=False)
parser.add_argument('--model_name', type=str, help='dataset selection')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_tasks', type=int, default=4, help='task number: chexpert:4, mura:1, hipxray:1')
parser.add_argument('--num_classes', type=int, default=14, help='task number: chexpert:14, mura:2, hipxray:2')
parser.add_argument('--view_num', type=int, default=2, help='view number')
parser.add_argument('--view_sel', type=int, default=1, help='view selection for chexpert dataset: 1: frontal 2:lateral')
parser.add_argument('--train_num', type=int, default=0, help='record the train number')
parser.add_argument('--epoch_num', type=int, default=0, help='record the model epoch')
parser.add_argument('--model_path', type=str, default='', help='Path to saved model checkpoint')
parser.add_argument('--output_file', type=str, default=None, help='Path to save predictions. If not provided, will save to same directory as model with name predictions.csv')
args = parser.parse_args()

if args.output_file is None:
    output_file = args.model_path
else:
    output_file = args.output_file


def plot_roc_curves(y_true, y_pred, class_names, save_path):
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(str(save_path))
    plt.close()


def main():
    config = get_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name == 'twoviewxfmamba':
        model = twoviewxfmamba(in_channels=1, outputs=config['num_classes'], pretrained='/pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth')
    elif args.model_name == 'twoviewxfmamba_tiny':
        model = twoviewxfmamba(in_channels=1, outputs=config['num_classes'], pretrained='./pretrained/vmamba/vssm1_tiny_0230s_ckpt_epoch_264.pth', type='tiny')
        target_layer = model.final_conv
    elif args.model_name == 'twoviewxfmamba_base':
        model = twoviewxfmamba(in_channels=1, outputs=config['num_classes'], hidden_dim= 1024, pretrained='./pretrained/vmamba/vssm_base_0229_ckpt_epoch_237.pth', type='base')
    
    if args.view_num == 2:
        model.zero_grad()
        model.eval()
        model.to(device)
        wrapped_model = ModelWrapper(model, output_index=0)  # Adjust output_index if necessary
        wrapped_model.to(device)
        wrapped_model.eval()
        # cam = pytorch_gradcam(model=wrapped_model, target_layers=[target_layer])
    elif args.view_num == 1:
        model.zero_grad()
        model.eval()
        model.to(device)
        # cam = pytorch_gradcam(model=model, target_layers=[target_layer])

    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"\nLoaded model checkpoint from {args.model_path}")

    # load train datasets root
    train_root_dirs =[
                './chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)',
                './chexpertchestxrays-u20210408/CheXpert-v1.0 batch 3 (train 2)',
                './chexpertchestxrays-u20210408/CheXpert-v1.0 batch 4 (train 3)'
            ]
    train_csv_file = ['./chexpertchestxrays-u20210408/train_1.csv']

    train_loader, val_loader, test_loader = create_train_val_test_data_loader_chexpert(train_root_dirs, train_csv_file, config)
    

    all_preds = []
    all_labels = []

    class_names = (
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
        'Support Devices', 'No Finding',
    )
    # Initialize variables for timing
    total_inference_time = 0
    inference_times = []
    batch_sizes = []
    
    # Start timer for total inference
    total_start_time = time.time()
    
    for idx, (image1, image2, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing test images"):
        
        image1 = image1.to(device)
        image2 = image2.to(device)
        labels = labels.to(device)
        
        # Start batch inference timer
        batch_start_time = time.time()
        
        if args.view_num == 2:
            outputs = model(image1, image2)
        elif args.view_num == 1:
            if args.view_sel == 1:
                outputs = model(image1)
            elif args.view_sel == 2:
                outputs = model(image2)   

        # End batch inference timer
        batch_end_time = time.time()
        batch_inference_time = batch_end_time - batch_start_time
        
        # Record timing information
        total_inference_time += batch_inference_time
        inference_times.append(batch_inference_time)
        batch_sizes.append(image1.size(0))
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Extract the primary tensor


        probabilities = torch.sigmoid(outputs) 
        probabilities = probabilities.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        all_preds.append(probabilities[0]) 
        all_labels.append(labels[0])
        if 0:
            # Create GradCAM visualization
            plt.figure(figsize=(10, 10))
            image1_np = image1[0]
            image_np = image1_np.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to [0,1]

            # If the image is grayscale (single channel), convert it to RGB
            if image_np.shape[2] == 1:
                image_np = np.repeat(image_np, 3, axis=2)
            image_np = (image_np * 255).astype(np.uint8)

            if args.view_num == 2:
                combined_input = torch.cat((image1, image2), dim=1)  # Shape: [1, 2, 224, 224]
                combined_input = combined_input.to(device)               
            elif args.view_num == 1:
                if args.view_sel == 1:
                    combined_input = image1
                    combined_input = combined_input.to(device)  
                elif args.view_sel == 2:
                    combined_input = image2
                    combined_input = combined_input.to(device)  

            grayscale_cam = cam(combined_input)
            cam_image = grayscale_cam[0]

            cam_image = np.maximum(cam_image, 0)  # ReLU
            cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min() + 1e-8)  # Normalize
            cam_image = np.uint8(255 * cam_image)  # Convert to uint8

            cam_image = cv2.resize(cam_image, (image_np.shape[1], image_np.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
            
            heatmap = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)
            
            original_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            overlay = cv2.addWeighted(original_bgr, 0.7, heatmap, 0.3, 0)
            
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            
            plt.imshow(overlay_rgb)
            plt.title(f'GradCAM Visualization', fontsize=16, pad=20)
        
            text_str = ""
            for class_idx, class_name in enumerate(class_names):
                gt_label = labels[0][class_idx]
                pred_prob = probabilities[0][class_idx]
                text_str += f'{class_name} - GT: {gt_label:.0f}, Pred: {pred_prob:.3f}\n'
            
            plt.text(0.02, -0.1, text_str, transform=plt.gca().transAxes, 
                    verticalalignment='top', fontsize=12)
            plt.axis('off')

            save_path = Path(output_file).parent / f"epoch_{args.epoch_num}_result" / "gradcam" / f"{idx}_gradcam.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)  # Create gradcam subfolder if it doesn't exist
            plt.tight_layout()
            plt.savefig(str(save_path), bbox_inches='tight')
            plt.close()

    all_preds = np.array(all_preds)  # Shape: [num_samples, 14]
    all_labels = np.array(all_labels)  # Shape: [num_samples, 14]

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    median_inference_time = np.median(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    
    total_images = sum(batch_sizes)
    per_image_time = total_inference_time / total_images
    
    timing_save_dir = Path(output_file).parent / f"epoch_{args.epoch_num}_result"
    timing_save_dir.mkdir(parents=True, exist_ok=True)
    timing_save_path = timing_save_dir / "inference_timing.txt"
    
    with open(timing_save_path, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total inference time: {total_inference_time:.4f} seconds\n")
        f.write(f"Total time (including data loading): {total_elapsed_time:.4f} seconds\n")
        f.write(f"Total images processed: {total_images}\n")
        f.write(f"Per-image inference time: {per_image_time*1000:.4f} ms\n")
        f.write(f"Mean batch inference time: {mean_inference_time*1000:.4f} ms\n")
        f.write(f"Std dev batch inference time: {std_inference_time*1000:.4f} ms\n")
        f.write(f"Median batch inference time: {median_inference_time*1000:.4f} ms\n")
        f.write(f"Min batch inference time: {min_inference_time*1000:.4f} ms\n")
        f.write(f"Max batch inference time: {max_inference_time*1000:.4f} ms\n")
        f.write(f"FPS (frames per second): {1.0/per_image_time:.2f}\n")
    
    print(f"\nInference timing saved to {timing_save_path}")
    print(f"Average inference time per image: {per_image_time*1000:.4f} ms")
    print(f"FPS: {1.0/per_image_time:.2f}")
    
    scores = []
    for i, col in enumerate(class_names):
        try:
            score = roc_auc_score(all_labels[:, i], all_preds[:, i])
            print(f"{col}: {score:.3f}")
        except ValueError as e:
            print(f"{col}: ROC AUC could not be computed - {e}")
        scores.append(score)
        
    average_roc_auc = np.mean(scores)
    print(f"\nAverage ROC AUC over 14 classes: {average_roc_auc:.4f}")
    # Plot ROC curves
    roc_dir = Path(output_file).parent / f"epoch_{args.epoch_num}_result" / "roc_curves"
    roc_dir.mkdir(parents=True, exist_ok=True)
    roc_plot_path = roc_dir / 'roc_curves.png'
    plot_roc_curves(all_labels, all_preds, class_names, roc_plot_path)
    print(f"\nROC curves saved to {roc_plot_path}")
    
    roc_txt_path = roc_dir / "roc_scores.txt"
    with open(roc_txt_path, "w") as f:
        f.write(f"Average ROC AUC over {len(class_names)} classes: {average_roc_auc:.4f}\n\n")
        f.write("Per-class ROC AUC scores:\n")
        for i, col in enumerate(class_names):
            if np.isnan(scores[i]):
                f.write(f"{col}: could not be computed\n")
            else:
                f.write(f"{col}: {scores[i]:.4f}\n")

    print(f"AUC scores saved to {roc_txt_path}")

if __name__ == "__main__":
    main()