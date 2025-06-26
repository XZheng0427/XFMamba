import numpy as np
import torch
import torchvision.models
import time
import csv
import wandb
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()

    '''
    models choose：resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202. mobilenetV1, inceptionV1
    '''
    parser.add_argument("--model", type=str, default="resnet32", required=False)
    parser.add_argument("--dataset", type=str, default='cifar100', required=False)
    parser.add_argument("--gpu", type=str, default="0,1", required=False)
    parser.add_argument("--num_classes", type=int, default=100, required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    parser.add_argument("--num_workers", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=200, required=False)
    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument("--optim", type=str, default='adamw', required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--lrscheduler", type=str, default='reduce', required=False)
    parser.add_argument("--weight_decay", type=float, default=0.0005, required=False)
    parser.add_argument("--momentum", type=float, default=0.9, required=False)
    parser.add_argument("--gamma", type=float, default=0.1, required=False)
    parser.add_argument("--device", type=str, default='cuda', required=False)
    parser.add_argument("--topk", type=int, default=1, required=False)
    parser.add_argument("--eval_flag", type=bool, default=True, required=False)

    args = parser.parse_args()
    args.model = 'resnet32, resnet32, resnet32'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    return args


def train_one_epoch_ml(epoch,
                    models,
                    model_names,
                    train_dataloader,
                    optimizers,
                    criterion_ce,
                    criterion_kl,
                    train_losses,
                    args,
                    config,
                    ):

    all_labels = [[] for _ in range(len(models))]
    all_probs = [[] for _ in range(len(models))]
    
    with tqdm(total=len(train_dataloader), desc=f'{epoch}/{args.epochs}', postfix=dict, maxinterval=1.0) as pbar:

        for i_batch, batch in enumerate(train_dataloader):
            image1, image2, labels = batch
            image1 = image1.to(args.device)
            image2 = image2.to(args.device)
            
            if config['dataset'] == 'chexpert':
                labels = labels.to(config['device']).float()
            elif config['dataset'] == 'mura' or config['dataset'] == 'hipxray':
                labels = labels.to(config['device'])
            
            outputs = []

            # calculate the output of each model
            for m, model in enumerate(models):
                optimizers[m].zero_grad()
                if model_names[m] == 'efffusionmambav3_ml2':
                    output = model(image1, image2)
                elif model_names[m] == 'singlevieweffmodel_ml2':
                    output = model(image1)
                
                if isinstance(outputs, tuple):
                    output = output[0]  # Extract the primary tensor  
                
                outputs.append(output)
 
            # calculate the loss of each model
            for i in range(len(models)):
                ce_loss = criterion_ce(outputs[i], labels)
                
                kl_loss = 0.

                for j in range(len(models)):
                    if i != j:
                        kl_loss += criterion_kl(F.log_softmax(outputs[i], dim=1),
                                                F.softmax(outputs[j].detach(), dim=1))

                loss = ce_loss + kl_loss / (len(models) - 1)
                
                loss.backward()
                optimizers[i].step()
                
                _, preds = torch.max(outputs[i], 1) 
                acc = torch.sum(preds == labels.data)
                
                probs = torch.softmax(outputs[i], dim=1)[:, 1]
                all_probs[i].extend(probs.detach().cpu().numpy())
                all_labels[i].extend(labels.detach().cpu().numpy())

                # record the loss of the model
                train_losses[i].update(ce_loss.item(), kl_loss.item(), loss.item(), acc.item(), epoch)

                # show the loss of the model in the current step.
                pbar.set_postfix({
                    # 'model': i,
                    # 'ce_loss': train_losses[i].ce_loss / (i_batch + 1),
                    # 'kl_loss': train_losses[i].kl_loss / (i_batch + 1),
                    'loss': train_losses[i].loss / (i_batch + 1),
                    'correct': train_losses[i].acc / len(train_dataloader.dataset),
                })

            pbar.update(1)
    
    for i in range(len(models)):
        # Binary Classification
        all_labels_np = np.array(all_labels[i])
        all_probs_np = np.array(all_probs[i])
        roc_auc = roc_auc_score(all_labels_np, all_probs_np)
        print(f"Epoch {epoch} Model {i} ROC AUC: {roc_auc:.4f}")
        
        train_losses[i].update_auc(roc_auc)
    

@torch.no_grad()
def val(epoch, models, model_names, val_dataloader, criterion_ce, criterion_kl, val_losses, args, config):

    for i in range(len(models)):
        models[i].eval()

    val_all_labels = [[] for _ in range(len(models))]
    val_all_probs = [[] for _ in range(len(models))]
    
    with tqdm(total=len(val_dataloader), desc=f'Epoch [{epoch}/{args.epochs}]') as pbar:
        for i_batch, batch in enumerate(val_dataloader):
            image1, image2, labels = batch
            image1 = image1.to(args.device)
            image2 = image2.to(args.device)
            
            if config['dataset'] == 'chexpert':
                labels = labels.to(config['device']).float()
            elif config['dataset'] == 'mura' or config['dataset'] == 'hipxray':
                labels = labels.to(config['device'])
                
            outputs = []

            # calculate the output of each model
            for m, model in enumerate(models):
                if model_names[m] == 'efffusionmambav3_ml2':
                    output = model(image1, image2)
                elif model_names[m] == 'singlevieweffmodel_ml2':
                    output = model(image1)
                
                if isinstance(outputs, tuple):
                    output = output[0]  # Extract the primary tensor  
                
                outputs.append(output)
            
            # calculate the loss of each model
            for i in range(len(models)):
                ce_loss = criterion_ce(outputs[i], labels)
                kl_loss = 0.

                for j in range(len(models)):
                    if i != j:
                        kl_loss += criterion_kl(F.log_softmax(outputs[i], dim=1),
                                                F.softmax(outputs[j].detach(), dim=1))

                loss = ce_loss + kl_loss / (len(models) - 1)
                
                preds = torch.argmax(outputs[i], dim=1)  # [batch_size]
                acc = torch.sum(preds == labels.data)
                
                probs = torch.softmax(outputs[i], dim=1)[:, 1]
                val_all_probs[i].extend(probs.detach().cpu().numpy())
                val_all_labels[i].extend(labels.detach().cpu().numpy())

                # record the loss
                val_losses[i].update(ce_loss.item(), kl_loss.item(), loss.item(), acc.item(), epoch)

            pbar.update(1)

    for i in range(len(models)):
        # Binary Classification
        all_labels_np = np.array(val_all_labels[i])
        all_probs_np = np.array(val_all_probs[i])
        roc_auc = roc_auc_score(all_labels_np, all_probs_np)
        print(f"Epoch {epoch} Model {i} ROC AUC: {roc_auc:.4f}")

        val_losses[i].update_auc(roc_auc)
        


# def train(args, model_str, project_name_, SAVE_PATH, is_augmentation):
#     utils.set_random_seed(args.seed)
#     args.model = model_str
#     for i in range(10):
#         try:
#             # wandb.init(project='mutual_involution_learning', name=project_name_)
#             wandb.init(project="BHF-ECG-Challenge", name=project_name_, entity="zhengxiaoyu0427-queen-mary-university-of-london")
#             break
#         except:
#             print('wandb init failed')
#             continue


#     wandb.config.update(args, allow_val_change=True)

#     criterion_ce = torch.nn.CrossEntropyLoss()
#     criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')

#     model_names, \
#     models, \
#     train_dataloader, \
#     val_dataloader = utils.get_model_and_dataset(args)

#     optimizers = utils.get_optimizer(args, models, len(models))
#     schedulers = utils.get_scheduler(args, optimizers)

#     n_train_batch = len(train_dataloader)
#     n_train_data = len(train_dataloader.dataset)
#     n_val_batch = len(val_dataloader)
#     n_val_data = len(val_dataloader.dataset)
#     best_val_acc = [0.] * len(models)
#     best_train_acc = [0.] * len(models)

#     utils.show_config(args)

#     for epoch in range(args.epochs):
#         train_losses = []
#         val_losses = []

#         for i in range(len(models)):
#             train_losses.append(utils.Totoal_Meter(n_train_batch, n_train_data))
#             val_losses.append(utils.Totoal_Meter(n_val_batch, n_val_data))
#             schedulers[i].step(epoch)
#             models[i] = models[i].to(args.device).train()

#         # train
#         train_one_epoch(epoch, models, train_dataloader, optimizers,
#                               criterion_ce, criterion_kl, train_losses, args, is_augmentation)

#         # val
#         if args.eval_flag:
#             val(epoch, models, val_dataloader,
#                       criterion_ce, criterion_kl, val_losses, args)

#         # save and print
#         # print(f'Epoch: [{epoch}/{args.epochs}]')
#         for i in range(len(models)):
#             # lr = utils.get_lr(optimizers[i])
#             train_res = train_losses[i].get_avg()
#             val_res = val_losses[i].get_avg()

#             if val_res['acc'] > best_val_acc[i]:
#                 best_val_acc[i] = val_res['acc']
#             if train_res['acc'] > best_train_acc[i]:
#                 best_train_acc[i] = train_res['acc']
#                 # torch.save(models[i].state_dict(), os.path.join(SAVE_PATH, f'{i}_{model_names[i]}_best.pth'))

#             # print(
#             #     f"{i}: {model_names[i]} lr={lr}"
#             #     f" [train_ce_loss={train_res['ce_loss']:.4f} train_kl_loss={train_res['kl_loss']:.4f} train_loss={train_res['loss']:.4f} train_acc={train_res['acc']:.4f}]"
#             #     f" [val_ce_loss={val_res['ce_loss']:.4f} val_kl_loss={val_res['kl_loss']:.4f} val_loss={val_res['loss']:.4f} val_acc={val_res['acc']:.4f}]"
#             # )

#             # save the training information in the wandb
#             wandb.log({
#                 "{}_{}_{}_{}_ceLoss".format(i, model_names[i], args.dataset, 'train'): train_res['ce_loss'],
#                 "{}_{}_{}_{}_klLoss".format(i, model_names[i], args.dataset, 'train'): train_res['kl_loss'],
#                 "{}_{}_{}_{}_loss".format(i, model_names[i], args.dataset, 'train'): train_res['loss'],
#                 "{}_{}_{}_{}_acc".format(i, model_names[i], args.dataset, 'train'): train_res['acc'],
#                 "{}_{}_{}_{}_ceLoss".format(i, model_names[i], args.dataset, 'val'): val_res['ce_loss'],
#                 "{}_{}_{}_{}_klLoss".format(i, model_names[i], args.dataset, 'val'): val_res['kl_loss'],
#                 "{}_{}_{}_{}_loss".format(i, model_names[i], args.dataset, 'val'): val_res['loss'],
#                 "{}_{}_{}_{}_acc".format(i, model_names[i], args.dataset, 'val'): val_res['acc'],
#             },step=epoch)

#         with open(os.path.join(SAVE_PATH, 'mutual_result.csv'), 'w', encoding='utf-8', newline='') as f:
#             fileName = ['model_name', 'train_acc', 'val_acc']
#             writer = csv.DictWriter(f, fieldnames=fileName)
#             writer.writeheader()
#             for i in range(len(model_names)):
#                 writer.writerow({'model_name':model_names[i],
#                                  'train_acc': best_train_acc[i],
#                                  'val_acc': best_val_acc[i],})

#     wandb.finish()
    

# def mutual_(model_str, proj_name, SAVE_PATH, is_augmentation):
#     args = parse_args()
#     train(args, model_str, proj_name, SAVE_PATH, is_augmentation=is_augmentation)


def save_best_roc_model_auc(model, optimizer, scheduler, epoch, val_roc, model_name, save_model_path):
    
    save_model_path = Path(save_model_path) if isinstance(save_model_path, str) else save_model_path
    
    model_dir = save_model_path / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f'best_model_{epoch}.pth'
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_roc': val_roc,
    }, save_path)
    
    print(f'\nSaved best model {model_name} with validation roc: {val_roc:.4f}')
     
    
    return save_path


if __name__ == '__main__':
    print('test')
    # mutual_()




