import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from tqdm import tqdm
from libs.dataset_chexpert_twoview import CheXpertDataset

def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(all_labels, all_preds, all_probs, epoch, dataset_name, config):
    epoch_folder = os.path.join(config['confusion_matrix_folder'], f"epoch_{epoch}")
    os.makedirs(epoch_folder, exist_ok=True)

    if dataset_name == 'mura' or dataset_name == 'hipxray':
        class_labels = [0, 1]  # Replace with your actual class labels

        cm = confusion_matrix(all_labels, all_preds, labels=class_labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

        disp.plot(cmap=plt.cm.Blues)

        plt.title(f'Confusion Matrix for {dataset_name} at Epoch {epoch}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        path_cm = os.path.join(epoch_folder, f"confusion_matrix_{dataset_name}_epoch_{epoch}.png")
        plt.savefig(path_cm)
        plt.close()

        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0  # Recall
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
        roc_auc = roc_auc_score(all_labels, all_probs)
        metrics_text = (
            f"Epoch {epoch} Metrics for {dataset_name}:\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Sensitivity (Recall): {sensitivity:.4f}\n"
            f"Specificity: {specificity:.4f}\n"
            f"F1 Score: {f1_score:.4f}\n"
            f"ROC-AUC: {roc_auc:.4f}\n"
        )

        path_metrics = os.path.join(epoch_folder, f"metrics_{dataset_name}_epoch_{epoch}.txt")
        with open(path_metrics, 'w') as f:
            f.write(metrics_text)
            
    elif dataset_name == 'chexpert':
        num_labels = config['num_classes']
        label_names = CheXpertDataset.LABELS  # Ensure this is accessible or pass as a parameter
        class_labels = [0, 1]  # Updated class labels for binary classification

        for i in range(num_labels):
            label_name = label_names[i]
            y_true = all_labels[:, i]
            y_pred = all_preds[:, i]
            y_prob = all_probs[:, i]  # Use probabilities for ROC AUC

            # Ensure labels are binary (0 or 1)
            y_true = y_true.astype(int)
            y_pred = y_pred.astype(int)

            if len(y_true) == 0:
                print(f"No valid samples for label '{label_name}' to plot confusion matrix.")
                continue

            # Diagnostic: Print number of positives and negatives
            num_positives = np.sum(y_true == 1)
            num_negatives = np.sum(y_true == 0)
            print(f"Label '{label_name}' - Positives: {num_positives}, Negatives: {num_negatives}")

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=class_labels)

            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
            disp.plot(cmap=plt.cm.Blues)

            plt.title(f'Confusion Matrix for {label_name} ({dataset_name}) at Epoch {epoch}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            path_cm = os.path.join(epoch_folder, f"confusion_matrix_{dataset_name}_{label_name}_epoch_{epoch}.png")
            plt.savefig(path_cm)
            plt.close()

            # Extract TN, FP, FN, TP from the confusion matrix
            if cm.shape == (2, 2):
                TN, FP, FN, TP = cm.ravel()
            else:
                print(f"Confusion matrix for label '{label_name}' is not 2x2.")
                TN, FP, FN, TP = 0, 0, 0, 0

            # Metrics calculation
            total = TN + FP + FN + TP
            accuracy = (TP + TN) / total if total != 0 else 0
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0  # Sensitivity
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            # ROC AUC calculation using predicted probabilities
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                roc_auc = None  # Undefined ROC AUC

            # Save metrics
            metrics_text = (
                f"Epoch {epoch} Metrics for {label_name} ({dataset_name}):\n"
                f"Accuracy: {accuracy:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall (Sensitivity): {recall:.4f}\n"
                f"Specificity: {specificity:.4f}\n"
                f"F1 Score: {f1_score:.4f}\n"
                f"ROC-AUC: {roc_auc if roc_auc is not None else 'Undefined'}\n"
            )

            path_metrics = os.path.join(epoch_folder, f"metrics_{dataset_name}_{label_name}_epoch_{epoch}.txt")

            with open(path_metrics, 'w') as f:
                f.write(metrics_text)



def train_one_epoch(model, train_loader, criterion, optimizer, config):
    # Training Phase
    model.train()  
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc="Training", leave=False):
        image1, image2, labels = batch
        image1 = image1.to(config['device'])
        image2 = image2.to(config['device'])
        
        # image1_np = image1[0]
        # image_np = image1_np.permute(1, 2, 0).cpu().numpy()
        
        # plt.figure(figsize=(10, 10))
    
        # # Display the image
        # plt.imshow(image_np, cmap='gray')
        # plt.axis('off')
        
        # # Define the filename, e.g., based on the index
        # save_root = '/data/DERI-USMSK/XiaoyuZheng-USMSK/cross-view-transformers-main/results/test_debug'
        # filename = os.path.join(save_root, f'image_1.png')
        
        # # Save the figure
        # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        
        # # Close the figure
        # plt.close()
        
        # # Optional: Print a message or perform other operations
        # print(f"Saved {filename}")
        
        # exit(0)
        
        if config['dataset'] == 'chexpert':
            labels = labels.to(config['device']).float()
        elif config['dataset'] == 'mura' or config['dataset'] == 'hipxray' or config['dataset'] == 'ddsmxray':
            labels = labels.to(config['device'])

        optimizer.zero_grad()
        if config['view_num'] == 1:
            if config['view_sel'] == 1: # Frontal view
                outputs = model(image1)
            elif config['view_sel'] == 2:
                outputs = model(image2) # Lateral view
        elif config['view_num'] == 2:
            outputs = model(image1, image2)  # Assuming model takes two images as input

        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Extract the primary tensor

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if config['num_tasks'] == 1:
            running_loss += loss.item() * image1.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1) 
            corrects = torch.sum(preds == labels.data).item()
            running_corrects += corrects
            all_preds.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
        else:
            # Multi-task for chexpert dataset
            running_loss += loss.item() * labels.numel()
            probs = torch.sigmoid(outputs)  
            preds = (probs >= 0.5).float()  
            corrects = torch.sum(preds == labels).item()
            running_corrects += corrects
            total_samples += labels.numel()
            all_preds.append(probs.detach().cpu().numpy())  # Shape: [batch_size, num_labels]
            all_labels.append(labels.detach().cpu().numpy())

   
    if config['num_tasks'] == 1:
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        all_preds = np.array(all_preds)  # Shape: [num_samples, num_labels]
        all_labels = np.array(all_labels)  # Shape: [num_samples, num_labels]

        epoch_roc_auc = roc_auc_score(all_labels, all_preds)
    else:
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples

        all_preds = np.concatenate(all_preds, axis=0)  # Shape: [num_samples, num_labels]
        all_labels = np.concatenate(all_labels, axis=0)  # Shape: [num_samples, num_labels]

        aucs = []
        for i in range(config['num_classes']):
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)
        
        epoch_roc_auc = np.nanmean(aucs)

    return epoch_loss, epoch_acc, epoch_roc_auc


def validator(model, val_loader, criterion, epoch, earlystop, config):
    # Validation Phase
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for image1, image2, labels in tqdm(val_loader, desc="Validation", leave=False):
            image1 = image1.to(config['device'])
            image2 = image2.to(config['device'])
            if config['dataset'] == 'chexpert':
                labels = labels.to(config['device']).float()
            elif config['dataset'] == 'mura' or config['dataset'] == 'hipxray' or config['dataset'] == 'ddsmxray':
                labels = labels.to(config['device'])

            if config['view_num'] == 1:
                if config['view_sel'] == 1: # Frontal view
                    outputs = model(image1)
                elif config['view_sel'] == 2:
                    outputs = model(image2) # Lateral view
            elif config['view_num'] == 2:
                outputs = model(image1, image2)  # Assuming model takes two images as input
                
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Extract the primary tensor

            loss = criterion(outputs, labels)

            if config['num_tasks'] == 1:
                val_running_loss += loss.item() * image1.size(0)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)  # [batch_size]
                corrects = torch.sum(preds == labels.data).item()
                val_running_corrects += corrects
                all_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
            else:
                # Multi-task multi-class
                val_running_loss += loss.item() * labels.numel()

                # Compute probabilities
                probs = torch.sigmoid(outputs)  # Convert logits to probabilities
                preds = (probs >= 0.5).float()

                # Compute number of correct predictions
                corrects = torch.sum(preds == labels).item()
                val_running_corrects += corrects
                total_samples += labels.numel()

                # Collect predictions and labels
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())    # Shape: [batch_size, num_labels]
                all_labels.append(labels.cpu().numpy())  # Shape: [batch_size, num_labels]

    if config['num_tasks'] == 1:
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects / len(val_loader.dataset)

        all_preds = np.array(all_preds)  # Shape: [num_samples, num_labels]
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)  # Shape: [num_samples, num_labels]
        val_epoch_roc_auc = roc_auc_score(all_labels, all_probs)
    else:
        val_epoch_loss = val_running_loss / total_samples
        val_epoch_acc = val_running_corrects / total_samples

        # Concatenate all predictions and labels
        all_preds = np.concatenate(all_preds, axis=0) 
        all_probs = np.concatenate(all_probs, axis=0)    # Shape: [num_samples, num_labels]
        all_labels = np.concatenate(all_labels, axis=0)  # Shape: [num_samples, num_labels]

        aucs = []
        for i in range(config['num_classes']):
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            aucs.append(auc)

        val_epoch_roc_auc = np.nanmean(aucs)

    earlystop(val_epoch_loss, model, epoch=epoch)

    # Plot confusion matrix (adjust as needed for multi-task multi-class)
    plot_confusion_matrix(all_labels, all_preds, all_probs, epoch, config['dataset'], config)

    return val_epoch_loss, val_epoch_acc, val_epoch_roc_auc, earlystop.early_stop


