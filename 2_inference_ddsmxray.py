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
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, flop_count_table
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import cv2
from libs.config import get_config
from libs.dataset_ddsmxray_twoview import create_ddsm_test_data_loader, create_ddsm_test_data_loader2
from pytorch_grad_cam import GradCAM as pytorch_gradcam
from net_fusionmamba import ModelWrapper
from net_fusionmamba import TwoViewXFMambaTop as twoviewxfmamba
from net_fusionmamba import SingleViewMamba as singleviewmamba
from net_fusionmamba import TwoViewLateJoinMamba as twoviewjoinmamba
from mvswintransformermodels.mvswintransformer import MVSwinTransformer as mvswintransformer

print("\n\n\n############################################################")
print("STARTING INFERENCE 🔍")
print(f"Python: {sys.version}\nPytorch: {torch.__version__}\nPytorch on GPU: {torch.cuda.is_available()}")
print("############################################################\n\n\n")

"""🔧 Parse command line arguments 🔧"""
def parse_option():
    parser = argparse.ArgumentParser('Cross-fusion Mamba training and evaluation script', add_help=False)
    parser.add_argument('--debug', action='store_true', default= False, help='Enable debug mode')
    parser.add_argument('--root_dir', type=str, default='.', help='Path to data directory')
    parser.add_argument('--train_image_paths', type=str, default='train_image_paths.csv', help='Path to train image')
    parser.add_argument('--train_image_labels', type=str, default='train_labeled_studies.csv', help='Path to train image labels')
    parser.add_argument('--valid_image_paths', type=str, default='valid_image_paths.csv', help='Path to validation image')
    parser.add_argument('--valid_image_labels', type=str, default='valid_labeled_studies.csv', help='Path to validation image labels')
    parser.add_argument('--body_parts', nargs='+', default=['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST'], help='Select body parts to test')
    parser.add_argument('--confusion_matrix_folder', type=str, default='output_confusionmatrix_chexpert_fusionattenv1')
    parser.add_argument('--dataset', type=str, default='mura', help='dataset selection')
    parser.add_argument("--epochs", type=int, default=100, required=False)
    parser.add_argument('--model_name', type=str, help='dataset selection')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_tasks', type=int, default=4, help='task number: chexpert:4, mura:1, hipxray:1, ddsmxray:1') 
    parser.add_argument('--num_classes', type=int, default=2, help='task number: chexpert:14, mura:2, hipxray:2, ddsmxray:2')
    parser.add_argument('--view_num', type=int, default=2, help='view number')
    parser.add_argument('--view_sel', type=int, default=1, help='view selection for chexpert dataset: 1: frontal 2:lateral')
    parser.add_argument('--train_num', type=int, default=0, help='record the train number')
    parser.add_argument('--epoch_num', type=int, default=0, help='record the model epoch')
    parser.add_argument('--model_path', type=str, default='', help='Path to saved model checkpoint')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save predictions. If not provided, will save to same directory as model with name predictions.csv')
    parser.add_argument('--check_mutual_learning', type=int, default=0, help='record the model epoch')
    parser.add_argument('--cal_flops', type=int, default=0, help='calculate the flops')
    args = parser.parse_args()
    config = get_config(args)

    return args, config


def plot_roc_curves(y_true, y_pred, save_path):
    plt.figure(figsize=(10, 8))
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def get_class_name(class_idx):
    class_mapping = {
        0: 'Negative',
        1: 'Positive',
        # Add more classes as needed
    }
    return class_mapping.get(class_idx, 'Unknown')

def main():
    args, config = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_file is None:
        output_file = args.model_path
    else:
        output_file = args.output_file

    if  args.model_name == 'twoviewxfmamba':
        model = twoviewxfmamba(in_channels=1, outputs=config['num_classes'], pretrained='./pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth')
        target_layer = model.final_conv
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
        cam = pytorch_gradcam(model=wrapped_model, target_layers=[target_layer])
    elif args.view_num == 1:
        model.zero_grad()
        model.eval()
        model.to(device)
        cam = pytorch_gradcam(model=model, target_layers=[target_layer])

    if args.cal_flops == 1:
        dummy_input1 = torch.randn(1, 1, 224, 224).to(device)
        dummy_input2 = torch.randn(1, 1, 224, 224).to(device) 
        flops = FlopCountAnalysis(model, (dummy_input1, dummy_input2))

        print("\nFLOPs Analysis:")
        print(flop_count_table(flops))
        print(f"Total GFLOPs: {flops.total() / 1e9:.2f}")
        exit(0)
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if args.check_mutual_learning == 0:
        model.load_state_dict(checkpoint)
    elif args.check_mutual_learning == 1:
        model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded model checkpoint from {args.model_path}")

    test_main_csv_files = [
            "./DDSMXray/calc_case_description_test_set.csv",
            "./DDSMXray/mass_case_description_test_set.csv"
    ]
    metadata_csv = "./DDSMXray/metadata.csv"
    base_dir = "./DDSMXray"

    results = {}
    combined_preds = []
    combined_labels = []
        
    test_loader = create_ddsm_test_data_loader(test_main_csv_files, metadata_csv, base_dir, config, crop_size=1, rescale_factor=None, verbose=True)
    test_loader2 = create_ddsm_test_data_loader2(test_main_csv_files, metadata_csv, base_dir, config, crop_size=1, rescale_factor=None, verbose=True)
    all_preds = []
    all_labels = []
    
    # Initialize variables for timing
    total_inference_time = 0
    inference_times = []
    batch_sizes = []
    
    # Start timer for total inference
    total_start_time = time.time()
    
    for idx, ((image1, image2, labels), (image1_2, image2_2, labels2)) in tqdm(
        enumerate(zip(test_loader, test_loader2)), 
        total=len(test_loader), 
        desc="Processing test images"
    ):      
        image1 = image1.to(device)
        image2 = image2.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():  
            if args.view_num == 2:
                outputs = model(image1, image2)
            elif args.view_num == 1:
                if args.view_sel == 1:
                    outputs = model(image1)
                elif args.view_sel == 2:
                    outputs = model(image2)   
       

        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Extract the primary tensor

        _, preds = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability for class 1 (e.g., 'Abnormal')
        probabilities = probabilities.detach().cpu().numpy()
        if 0:
            # Create GradCAM visualization
            plt.figure(figsize=(10, 10))
            
            image1_np = image1_2[0]
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
            text_str = f'GT:{get_class_name(labels.item())} - Pred: {get_class_name(preds.item())} ({probabilities.item():.2f})'
            plt.text(0.02, -0.1, text_str, transform=plt.gca().transAxes, 
                    verticalalignment='top', fontsize=12)
            plt.axis('off')

            save_path = Path(output_file).parent / f"epoch_{args.epoch_num}_result"/ f"gradcam" / f"{idx}_gradcam.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)  # Create gradcam subfolder if it doesn't exist
            plt.tight_layout()
            plt.savefig(str(save_path), bbox_inches='tight')
            plt.close()

        all_preds.append(probabilities.item())  # Probability for ROC curve
        all_labels.append(labels.item())

        combined_preds.append(probabilities.item())
        combined_labels.append(labels.item())


    if len(np.unique(combined_labels)) >= 2:
        combined_roc_auc = roc_auc_score(combined_labels, combined_preds)
        combined_fpr, combined_tpr, _ = roc_curve(combined_labels, combined_preds)
        plt.plot(combined_fpr, combined_tpr, label=f'Combined (AUC = {combined_roc_auc:.4f})', linewidth=6, linestyle='-')
    else:
        print("Skipping combined ROC curve due to insufficient classes.")
    # Configure plot
    # plt.xlabel('False Positive Rate', fontsize=26)  # Adjust font size
    # plt.ylabel('True Positive Rate', fontsize=26)  # Adjust font size
    # plt.xticks(fontsize=24)  # Adjust tick label font size
    # plt.yticks(fontsize=24)  # Adjust tick label font size
    # # plt.title('ROC Curves for All Body Parts')
    # plt.legend(loc='lower right', fontsize=24)  # Adjust legend font size

    # Define save path for ROC curves
    roc_save_dir = Path(output_file).parent / f"epoch_{args.epoch_num}_result"/"roc_curves"
    roc_save_dir.mkdir(parents=True, exist_ok=True)
    roc_save_path = roc_save_dir / "ddsmxray_test_roc_curves.png"

    # Save the ROC curves plot
    # plt.tight_layout()
    # plt.savefig(str(roc_save_path), bbox_inches='tight')
    # plt.close()
    plot_roc_curves(all_labels, all_preds, roc_save_path)
    
    print(f"\nROC curves saved to {roc_save_path}")
    print(f'Combined ROC AUC Score: {combined_roc_auc if len(np.unique(combined_labels)) >= 2 else "N/A"}')

    roc_scores_path = roc_save_dir / "ddsmxray_test_roc_scores.txt"
    with open(roc_scores_path, "w") as f:
        if combined_roc_auc is not None:
            f.write(f"Combined ROC AUC: {combined_roc_auc:.4f}\n")
        else:
            f.write("Combined ROC AUC: N/A\n")

    print(f"AUC scores saved to {roc_scores_path}")
    
if __name__ == "__main__":
    main()