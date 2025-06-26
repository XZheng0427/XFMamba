import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.optim.lr_scheduler import StepLR
import argparse 
import wandb
from tqdm import tqdm 
from net_fusionmamba import TwoViewXFMambaTop as twoviewxfmamba

from libs.config import get_config
from libs.dataset_mura_twoview import create_data_loader4 as create_data_loader_mura4
from libs.dataset_chexpert_twoview import create_train_val_test_data_loader as create_train_val_test_data_loader_chexpert
from libs.dataset_ddsmxray_twoview import create_ddsmxray_data_loader as create_train_val_data_loader_ddsmxray
from libs.dataset_hipxray_twoview import create_data_loader as create_data_loader_hipxray
from libs.training import set_random_seeds, train_one_epoch, validator
from early_stop import EarlyStopping

print("\n\n\n############################################################")
print("LET'S START TRAINING 👍👍👍👍👍👍👍👍👍👍👍👍")
print(f"Python: {sys.version}\nPytorch: {torch.__version__}\nPytorch on GPU: {torch.cuda.is_available()}")
print("############################################################\n\n\n")

def parse_option():
    parser = argparse.ArgumentParser('Cross-fusion Mamba training and evaluation script', add_help=False)
    parser.add_argument('--debug', action='store_true', default= False, help='Enable debug mode')
    parser.add_argument('--root_dir', type=str, default='.', help='Path to data directory')
    parser.add_argument('--train_image_paths', type=str, default='train_image_paths.csv', help='Path to train image')
    parser.add_argument('--train_image_labels', type=str, default='train_labeled_studies.csv', help='Path to train image labels')
    parser.add_argument('--valid_image_paths', type=str, default='valid_image_paths.csv', help='Path to validation image')
    parser.add_argument('--valid_image_labels', type=str, default='valid_labeled_studies.csv', help='Path to validation image labels')
    parser.add_argument('--body_parts', default=['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST'], help='select body parts to train')
    parser.add_argument('--confusion_matrix_folder', type=str, default='output_confusionmatrix_chexpert_fusionattenv1')
    parser.add_argument('--savemodel_path', type=str, help='Path to save model')
    parser.add_argument('--pretrained_model_path', type=str, default='./pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth', help='pre-trained model path')
    parser.add_argument('--dataset', type=str, default='chexpert', help='dataset selection')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_name', type=str, help='dataset selection')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_tasks', type=int, default=4, help='task number: chexpert:4, mura:1, hipxray:1')
    parser.add_argument('--num_classes', type=int, default=14, help='task number: chexpert:14, mura:2, hipxray:2, ddsmxray:2')
    parser.add_argument('--view_num', type=int, default=2, help='view number')
    parser.add_argument('--view_sel', type=int, default=1, help='view selection for chexpert dataset: 1: frontal 2:lateral')
    parser.add_argument('--train_num', type=int, default=0, help='record the train number')
    parser.add_argument("--epochs", type=int, default=100, required=False)
    parser.add_argument('--wandb', type=bool, default=False, help='Enable wandb')
    args = parser.parse_args()
    config = get_config(args)

    return args, config

if __name__ ==  "__main__":
    args, config = parse_option()

    if args.wandb == True:
        if args.dataset == 'mura':
            wandb.init(project=".", entity=".",  name=f'train_{args.dataset}_{args.model_name}_{args.train_num}')
        elif args.dataset == 'chexpert':
            wandb.init(project=".", entity=".",  name=f'train_{args.dataset}_{args.model_name}_{args.train_num}')
        elif args.dataset == 'ddsmxray':
            wandb.init(project=".", entity=".",  name=f'train_{args.dataset}_{args.model_name}_{args.train_num}')

        wandb.config.update(config)

    set_random_seeds(config['seed'])

    print(f"Strat processing {args.dataset} dataset")
    if args.dataset == 'mura':
        train_image_paths_csv = os.path.join(args.root_dir, 'MURA-v1.1', args.train_image_paths)
        train_study_labels_csv = os.path.join(args.root_dir, 'MURA-v1.1', args.train_image_labels)
        valid_image_paths_csv = os.path.join(args.root_dir, 'MURA-v1.1', args.valid_image_paths)
        valid_study_labels_csv = os.path.join(args.root_dir, 'MURA-v1.1', args.valid_image_labels)


        train_loader, val_loader, test_loader = create_data_loader_mura4(train_image_paths_csv, 
                                                                         train_study_labels_csv, 
                                                                         valid_image_paths_csv, 
                                                                         valid_study_labels_csv, 
                                                                         args.body_parts, 
                                                                         config, 
                                                                         save_csv=False,
                                                                         save_dir='/data/DERI-USMSK/MURA-v1.1')
  
    elif args.dataset == 'chexpert':
        if args.debug == True:
            train_root_dirs =[
            '/data/DERI-USMSK/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 4 (train 3)'
            ]
        else:
            train_root_dirs =[
                '/data/DERI-USMSK/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)',
                '/data/DERI-USMSK/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 3 (train 2)',
                '/data/DERI-USMSK/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 4 (train 3)'
            ]

        train_csv_file = ['/data/DERI-USMSK/chexpertchestxrays-u20210408/train_1.csv']

        # train_loader, val_loader = create_data_loader_chexpert(train_root_dirs, train_csv_file, config)
        train_loader, val_loader, test_loader = create_train_val_test_data_loader_chexpert(train_root_dirs, train_csv_file, config)

    elif args.dataset == 'ddsmxray':
        main_csv_files = [
            "/data/DERI-USMSK/DDSMXray/calc_case_description_train_set.csv",
            "/data/DERI-USMSK/DDSMXray/mass_case_description_train_set.csv"
        ]
        metadata_csv = "/data/DERI-USMSK/DDSMXray/metadata.csv"
        base_dir = "/data/DERI-USMSK/DDSMXray"
        train_loader, val_loader = create_train_val_data_loader_ddsmxray(main_csv_files, metadata_csv, base_dir, config, crop_size=1, verbose=False)
            
    elif args.dataset == 'hipxray':
        images_dir = '/data/DERI-USMSK/XavierHipXray/Images'
        csv_file = '/data/DERI-USMSK/XavierHipXray/hipxray-label.csv'
        train_loader, val_loader = create_data_loader_hipxray(images_dir, csv_file, config)
        
    # model = fusionmambav2(in_channels=1, outputs=config['num_classes'], pretrained=True, dropout=False, attention_combine='add')

    if args.model_name == 'twoviewxfmamba':
        model = twoviewxfmamba(in_channels=1, depth=2, outputs=config['num_classes'], pretrained=args.pretrained_model_path)
    elif args.model_name == 'twoviewxfmamba_tiny':
        model = twoviewxfmamba(in_channels=1, outputs=config['num_classes'], pretrained=args.pretrained_model_path, type='tiny')
    elif args.model_name == 'twoviewxfmamba_base':
        model = twoviewxfmamba(in_channels=1, outputs=config['num_classes'], hidden_dim= 1024, pretrained=args.pretrained_model_path, type='base')
    
    print(f'model name: {args.model_name}')

    model.to(config['device'])


    if args.dataset == 'mura' or args.dataset == 'hipxray' or args.dataset == 'ddsmxray':
        criterion = nn.CrossEntropyLoss()
    elif args.dataset == 'chexpert':
        criterion = nn.BCEWithLogitsLoss()

    
    # optimizer = optim.Adam(model.parameters(), betas=config['betas'], lr=config['learning_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)  
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=config['learning_rate'],
    #     weight_decay=config['weight_decay']
    # )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.05,
    #     patience=5,
    #     verbose=True
    # )

    earlystop = EarlyStopping(patience=config['early_stopping_patience'], verbose=True, path=args.savemodel_path)

    for epoch in tqdm(range(config['num_epochs']), desc='Training epochs'):     
        # Training phase
        epoch_loss, epoch_acc, epoch_roc_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, config
        )

        # Validation phase
        val_epoch_loss, val_epoch_acc, val_epoch_roc_auc, early_stop = validator(
            model, val_loader, criterion, epoch, earlystop, config
        )

        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val   Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        # Logging metrics
        if args.wandb == True:
            wandb.log({
                'train_loss': epoch_loss,
                'val_loss': val_epoch_loss,
                'train_roc': epoch_roc_auc,
                'val_roc': val_epoch_roc_auc,
                'train_acc': epoch_acc,
                'val_acc': val_epoch_acc
            })

        if early_stop:
            print("Early stopping triggered.")
            break
        
        scheduler.step()

    print('Training completed.')
