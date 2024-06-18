import torch
import yaml
import argparse
import time
import sys
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

#--
sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace")
import map_dataset
import map_train

sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Libs")
import RS_dataset
import RS_models
import RS_utils

import lightning as L
#from lightning.fabric import Fabric

#--- Console 
console = Console(record=True)

#--- load for map your city 
#--- path
input_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/Map_Your_City/building-age-dataset/"
train_path = input_path + "train/data/"
test_path = input_path + "test/data/"

#--- Load csv files
train_df = pd.read_csv(input_path + "train/train-set.csv")
test_df = pd.read_csv(input_path + "test/test-set.csv") 

#--- data split 
names_data = os.listdir(train_path)
#parse_idx = int(len(names_data) * 0.005)
#names_data = names_data[:parse_idx]

names_train, names_valid = train_test_split(names_data, test_size=0.1, random_state=1)



def main(cfg: argparse.Namespace):
    #--- logger 
    log_path = f'/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace/logs/ver_{cfg.MODEL}_%Y-%m-%d_%H-%M-%S.log'
    logger = RS_utils.log_creator(log_path)

    #--- save config
    start = time.time()
    save_dir = Path(cfg.SAVE_DIR)
    save_dir.mkdir(exist_ok=True)
    fix_seeds(123)
    setup_cudnn()

    #--- parallel 
    fabric = L.fabric.Fabric(accelerator="cuda", devices=4, strategy="ddp")
    fabric.launch()
    
    #--- device
    device = torch.device(cfg.DEVICE)
    print('Device: ' + str(device))
    num_workers = 8   

    #--- model 
    model = timm.create_model(
    cfg.MODEL,
    pretrained=True,
    num_classes=cfg.CLASSES_NUM )  # remove classifier nn.Linear
    
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    data_transform = timm.data.create_transform(**data_config, is_training=False)
    # check the resize & normalization option for model optimization 
    print("# data config ------------------ : ", data_config)
    print("# data transform ------------------ : ", data_transform) 
    

    #--- train and valid selection need
    train_set = map_dataset.Map_Dataset(names_train,data_transform,train_path) 
    valid_set = map_dataset.Map_Dataset(names_valid,data_transform,train_path)  

    #--- dataloader 
    trainloader = DataLoader(train_set, cfg.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    testloader = DataLoader(valid_set, cfg.BATCH_SIZE, num_workers=num_workers, pin_memory=True) 


    #--- freeze layers or not
    if cfg.FREEZE:
        for n, p in model.named_parameters():
            if 'head' not in n:
                p.requires_grad_ = False

    #--- loss functions
    train_loss_fn = LabelSmoothCrossEntropy(smoothing=0.1)
    val_loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), cfg.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    scaler = GradScaler(enabled=cfg.AMP)
    
    #--- score var
    best_top1_acc, best_top5_acc = 0.0, 0.0

    #--- fabric
    if fabric:
        model, optimizer = fabric.setup(model, optimizer)
        trainloader,testloader = fabric.setup_dataloaders(trainloader,testloader)


    #--- train & valid 
    #model = model.to(device)
    for epoch in range(cfg.EPOCHS):
        
        #--- train
        map_train.train(trainloader, model, train_loss_fn, optimizer, scheduler, scaler, device, epoch, cfg, logger,fabric)

        if (epoch+1) % cfg.EVAL_INTERVAL == 0 or (epoch+1) == cfg.EPOCHS:
    
            print ("Sleep 5 seconds from now on...")
            torch.cuda.empty_cache() 
            time.sleep(5)
            print("wake up!")
            
            #--- valid
            top1_acc, top5_acc = map_train.test(testloader, model, val_loss_fn, device, logger, cfg)

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top5_acc = top5_acc
                torch.save(model.state_dict(), save_dir / f"{cfg.MODEL}_epoch_{epoch}.pth")
            console.print(f" Best Top-1 Accuracy: [red]{(best_top1_acc):>0.1f}%[/red]\tBest Top-5 Accuracy: [red]{(best_top5_acc):>0.1f}%[/red]\n")
            log_dict_test = {
                "epoch" : epoch,
                "batch_size" : int(cfg.BATCH_SIZE),
                "lr" : optimizer.param_groups[0]['lr'],
                "best_top1_acc" : best_top1_acc,
                "best_top5_acc" : best_top5_acc
            }
            logger.info(log_dict_test)
    
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