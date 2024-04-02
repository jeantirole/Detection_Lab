

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from albumentations import *
from torchvision import transforms
from torch.utils.data import DataLoader
from glob import glob
import segmentation_models_pytorch as smp

import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm 

import wandb
import logging
from tqdm import tqdm
import datetime

#---
import dataset_rs
import utils_rs


#-- args
EXEC_VER = 88 
BATCH_SIZE = 8
DEVICE = "cuda:0"
img_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/01.512_imgs"
mask_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/02.512_masks"



#-- logger
# Set up logging
log_filename = datetime.datetime.now().strftime(f'./01.log/ver_{EXEC_VER}_%Y-%m-%d_%H-%M-%S.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.FileHandler(log_filename)
logger.addHandler(handler)

#-- datasets 
train_dataset = dataset_rs.Seg_RS_dataset(img_dir=img_path, mask_dir=mask_path,image_resize = 512, phase="train" )
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn)

valid_dataset = dataset_rs.Seg_RS_dataset(img_dir=img_path,mask_dir=mask_path,image_resize = 512, phase="val" )
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=valid_dataset.collate_fn)

data_loader= {}
data_loader["train"] = train_loader
data_loader["valid"] = valid_loader

#-- model 
model = smp.DeepLabV3Plus(
    encoder_name="resnet152",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=16,                      # model output channels (number of classes in your dataset)
)

model = model.to(DEVICE)

#-- loss 
#criterion = nn.CrossEntropyLoss()
criterion = dataset_rs.UNet_metric(num_classes=16)

#-- optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# run
epochs = 999
iteration = 0
total_loss = 0

for epoch in range(epochs):
    
    iteration = 0
    epoch_running_loss = 0 
    
    #tqdm_iterator = tqdm(data_loader["train"], desc=f"Epoch {epoch}")
    
    for index, data in enumerate(data_loader["train"]):
        
        imgs, masks = data
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        
        # opt
        optimizer.zero_grad()
        
        # runs
        outputs = model(imgs)
        
        # loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # stat
        epoch_running_loss += loss.item()

        # log
        logger.info(f"epoch : {epoch} iter : {index} loss: {loss:.8f}")
        log = {'loss': f'{loss / 10:.8f}' }
        #wandb.log(log)

        log_iter = 100
        
        if (index % log_iter) == 0:    # print every 2000 mini-batches
            print(f"epoch : {epoch} , iter : {index} , total_iter : {len(data_loader['train'])} , running_loss : {epoch_running_loss / (index +1)}")
        
    
    #-- save 
    save_path = f"./02.ckpts/ver_{EXEC_VER}_epoch_{epoch + 1}.pt"
    torch.save(model.state_dict(), save_path)