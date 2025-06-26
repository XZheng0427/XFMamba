from pathlib import Path
import torch

def get_config(args):
    """Get configuration dictionary for model training"""
    return {
        'root_dir': Path(args.root_dir),
        'train_image_paths': Path(args.train_image_paths),
        'train_image_labels': args.train_image_labels,
        'valid_image_paths': args.valid_image_paths,
        'valid_image_labels': args.valid_image_labels,
        'confusion_matrix_folder': args.confusion_matrix_folder,
        'dataset': args.dataset,
        'view_num': args.view_num,
        'view_sel': args.view_sel,
        'train_ratio': 0.85,
        'valid_ratio': 0.15,
        'image_size': (224, 224),
        'batch_size': args.batch_size,
        'num_workers': 4,
        'pin_memory': True,
        'seed': 42,
        'num_classes': args.num_classes, #14
        'num_tasks': args.num_tasks, #4
        'learning_rate': args.lr,  #1e-4,
        'weight_decay': 0.01,
        'betas':(0.9,0.999),
        'weight_decay': 0.01,
        'num_epochs': args.epochs,
        'early_stopping_patience': 100,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        
    }