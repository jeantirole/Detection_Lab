import torch
import yaml
import argparse
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from torchvision.datasets import *
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
import timm

#from datasets.transforms import get_train_transforms, get_val_transforms
#from models import *
from utils.losses import LabelSmoothCrossEntropy, CrossEntropyLoss
from utils.utils import fix_seeds, setup_cudnn, create_progress_bar
from utils.metrics import compute_accuracy

from torchvision.datasets import ImageFolder
from typing import Optional, Callable
from pathlib import Path

import torch
import os 
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
import cv2
import rasterio 
from urllib.request import urlopen
from PIL import Image

#-- Console 
console = Console()


#-- load for map your city 
#--- path
input_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/Map_Your_City/building-age-dataset/"
train_path = input_path + "train/data/"
test_path = input_path + "test/data/"

#--- device
device = torch.device("cuda:0")
print('Device: ' + str(device))   

#--- Load csv files
train_df = pd.read_csv(input_path + "train/train-set.csv")
test_df = pd.read_csv(input_path + "test/test-set.csv") 

#--- data split 
names_data = os.listdir(train_path)
names_train, names_valid = train_test_split(names_data, test_size=0.1, random_state=1)

#--- dataset class 
class Map_Dataset(torch.utils.data.Dataset):      
    def __init__(self, list_IDs,transform): 
        self.list_IDs = list_IDs
        self.transform = transform
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index): 
        ID = self.list_IDs[index] 
        X1 = Image.open(train_path + ID + '/street.jpg').convert('RGB')
        #X1 = cv2.resize(X1, (256, 256))
        #X1 = np.asarray(X1)
        #X1 = np.transpose(X1,[1, 2, 0])
        X1 = self.transform(X1)
        #X1 = cv2.resize(X1, (256, 256)) 
        
        #X2 = cv2.imread(train_path + ID + '/orthophoto.tif')  
        #X2 = cv2.resize(X2, (256, 256)) 
        
        #X3 = rasterio.open(train_path + ID + '/s2_l2a.tif').read() 
        #X3 = np.transpose(X3, [1, 2, 0]) 
        
        y = int(open(train_path + ID + '/label.txt', "r").read())
        return X1, y 
    
#---


def train(dataloader, model, loss_fn, optimizer, scheduler, scaler, device, epoch, cfg):
    model.train()
    progress = create_progress_bar()
    lr = scheduler.get_last_lr()[0]
    task_id = progress.add_task(description="", total=len(dataloader), epoch=epoch+1, epochs=cfg.EPOCHS, lr=lr, loss=0.)

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        with autocast(enabled=cfg.AMP):
            
            pred = model(X)
            loss = loss_fn(pred, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        progress.update(task_id, description="", advance=1, refresh=True, loss=loss.item())
    scheduler.step()
    progress.stop()


def test(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, top1_acc, top5_acc = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            acc1, acc5 = compute_accuracy(pred, y, topk=(1, 5))
            top1_acc += acc1 * X.shape[0]
            top5_acc += acc5 * X.shape[0]

    test_loss /= num_batches
    top1_acc /= size
    top5_acc /= size
    console.print(f"\n Top-1 Accuracy: [blue]{(top1_acc):>0.1f}%[/blue],\tTop-5 Accuracy: [blue]{(top5_acc):>0.1f}%[/blue],\tAvg Loss: [blue]{test_loss:>8f}[/blue]")
    return top1_acc, top5_acc

    

def main(cfg: argparse.Namespace):
    start = time.time()
    save_dir = Path(cfg.SAVE_DIR)
    save_dir.mkdir(exist_ok=True)
    fix_seeds(123)
    setup_cudnn()
    best_top1_acc, best_top5_acc = 0.0, 0.0
    
    device = torch.device(cfg.DEVICE)
    num_workers = 8

    #--- model 
    #model = get_model(model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',num_classes=7)
    model = timm.create_model(
    'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
    pretrained=True,
    num_classes=7 )  # remove classifier nn.Linear
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    data_transform = timm.data.create_transform(**data_config, is_training=True)

    # augmentations
    # train_transforms = get_train_transforms(cfg.IMAGE_SIZE)
    # val_transforms = get_val_transforms(cfg.EVAL_IMAGE_SIZE)
    # data_transform = transforms.Compose([  
    #     transforms.ToTensor() 
    # ])

    #--- train and valid selection need
    train_set = Map_Dataset(names_train,data_transform) 
    valid_set = Map_Dataset(names_valid,data_transform)  

    # create a dataset class
    #trainset = eval(cfg.DATASET)('data', train=True, transform=train_transforms, download=True)
    #testset = eval(cfg.DATASET)('data', train=False, transform=val_transforms, download=True)
    trainset = train_set
    testset = valid_set

    # dataloader 
    trainloader = DataLoader(trainset, cfg.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, cfg.BATCH_SIZE, num_workers=num_workers, pin_memory=True)

    # initialize model and load imagenet pretrained

    #model = eval(cfg.MODEL)(cfg.VARIANT, cfg.PRETRAINED, cfg.CLASSES_NUM, cfg.IMAGE_SIZE)
    # soft max need ! 

    # freeze layers or not
    if cfg.FREEZE:
        for n, p in model.named_parameters():
            if 'head' not in n:
                p.requires_grad_ = False

    model = model.to(device)
    train_loss_fn = LabelSmoothCrossEntropy(smoothing=0.1)
    val_loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), cfg.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    scaler = GradScaler(enabled=cfg.AMP)
    
    for epoch in range(cfg.EPOCHS):
        train(trainloader, model, train_loss_fn, optimizer, scheduler, scaler, device, epoch, cfg)

        if (epoch+1) % cfg.EVAL_INTERVAL == 0 or (epoch+1) == cfg.EPOCHS:
            top1_acc, top5_acc = test(testloader, model, val_loss_fn, device)

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top5_acc = top5_acc
                torch.save(model.state_dict(), save_dir / f"{cfg.MODEL}_{cfg.VARIANT}_epoch_{epoch}.pth")
            console.print(f" Best Top-1 Accuracy: [red]{(best_top1_acc):>0.1f}%[/red]\tBest Top-5 Accuracy: [red]{(best_top5_acc):>0.1f}%[/red]\n")
    
    end = time.gmtime(time.time() - start)
    total_time = time.strftime("%H:%M:%S", end)

    table = Table(show_header=True, header_style="magenta")
    table.add_column("Best Top-1 Accuracy")
    table.add_column("Best Top-5 Accuracy")
    table.add_column("Total Training Time")
    table.add_row(f"{best_top1_acc}%", f"{best_top5_acc}%", str(total_time))
    console.print(table)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace/configs/finetune.yaml')
    args = parser.parse_args()
    cfg = argparse.Namespace(**yaml.load(open(args.cfg), Loader=yaml.SafeLoader))
    main(cfg)