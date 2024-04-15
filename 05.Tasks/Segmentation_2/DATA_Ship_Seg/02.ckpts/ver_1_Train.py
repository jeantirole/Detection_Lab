import sys 
sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Libs")
import RS_utils
import RS_dataset
import RS_models

#---
import os
import shutil
from collections import defaultdict
import numpy as np 
from glob import glob
import os
import torch
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
#---
import torch.nn as nn 
import datetime
import logging
import easydict
#---
from lightning.fabric import Fabric
import lightning as L
import segmentation_models_pytorch as smp

#-- data
img_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/01.512_imgs"
mask_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/02.512_masks"

img_path_ship  = np.array(sorted(glob(os.path.join(img_path, "*.png"))) )
mask_path_ship = np.array(sorted(glob(os.path.join(mask_path, "*.png"))) )

aa = np.load("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Tasks/Segmentation_1/03.data_list/512_ships.npy")

selected_paths_img = img_path_ship[aa]
selected_paths_mask  = mask_path_ship[aa]

#---------------------------------
#-- args operation
EXEC_VER = 1 # zero means test 
DDP = False
TASK = "Seg"
#-- args data 
CROP_SIZE = 256
DATA_PATH_IMG = selected_paths_img
DATA_PATH_LABEL = selected_paths_mask
#-- args modeling 
MODEL_NAME = "Unet++"
MODEL_IN_CHANNEL = 3   # image channel  
MODEL_OUT_CHANNEL = 3  # mask channel and category 
CRITERION = "CE"
BATCH_SIZE = 8 # 4
LEARNING_RATE = 1e-3 #1e-5
#-- args train
DEVICE = "cuda:0"
DEVICES = [0,1,2,3]
RESUME = False
SAVE_EPOCH = 10
EPOCHS = 140
DATA_SHUFFLE = True
#-- args category 
ISAID_CLASSES_SHIP = (
    'background','ship','harbor' 
    )
ISAID_PALETTE_SHIP = {
    0: (0, 0, 0), 
    1: (0, 0, 63), 
    2: (0, 100, 155)
    }
#---------------------------------

#--- logger
log_path = f'/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Tasks/Segmentation_2/DATA_Ship_Seg/01.log/ver_{EXEC_VER}_%Y-%m-%d_%H-%M-%S.log'
logger = RS_utils.log_creator(log_path)


#-- dataset
train_dataset = RS_dataset.Seg_RS_dataset_edge_v5(img_dir = DATA_PATH_IMG, 
                                                  mask_dir = DATA_PATH_LABEL, 
                                                  image_resize = None, 
                                                  crop_size = CROP_SIZE, 
                                                  phase = "train",
                                                  palette=ISAID_PALETTE_SHIP,
                                                  edge_return=False,
                                                  mask_onehot=False)

train_dataloader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=DATA_SHUFFLE, collate_fn=train_dataset.collate_fn)

#--- model 
model = smp.UnetPlusPlus(encoder_name="resnet152",classes=MODEL_OUT_CHANNEL)


if CRITERION =="BCE":
    criterion = nn.BCELoss()
elif CRITERION == "CE":
    criterion = nn.CrossEntropyLoss(reduction="mean")   

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)


if DDP:
    #--- fabric setup 
    fabric = L.Fabric(accelerator="cuda", devices=DEVICES, strategy="ddp")
    fabric.launch()
    
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(train_dataloader)
else:
    model = model.to(DEVICE)

#--- fabric setup 
iteration = 0
for epoch in range(EPOCHS):
    
    epoch_running_loss = 0 
    
    for index, data in enumerate(train_dataloader):
        
        image, mask = data
    
        #---
        if DDP != True:    
            image, mask = image.to(DEVICE), mask.to(DEVICE)
        
        # opt
        optimizer.zero_grad()

        # runs
        outputs = model(image)
        
        #--- get dice 
        # loss
        #outputs = nn.functional.sigmoid(outputs)
        loss = criterion(outputs, mask)
        
        coef_dice = RS_utils.dice_coef(outputs, mask, n_class=MODEL_OUT_CHANNEL, batch=True, loss_type="CE")
        
        
        if DDP:
            fabric.backward(loss)
        else:
            loss.backward()
        
        optimizer.step()
        
        # stat
        epoch_running_loss += loss.item()
        
        log_dict=   {
            "epoch" : epoch,
            "iteration" : iteration,
            "loss" : "{:.5f}".format(loss.item()),  
            "dice" : "{:.5f}".format(coef_dice.item()),
            "lr" : optimizer.param_groups[0]['lr']
        }        
        logger.info(log_dict)

        iteration += index
    #-- epoch set scheduler
    scheduler.step()

    #-- save 
    
    if epoch % SAVE_EPOCH ==0:
        #-- weight save 
        save_path = f"./02.ckpts/ver_{EXEC_VER}_{TASK}_{MODEL_NAME}_epoch_{epoch}_iteration_{iteration}.pt"
        torch.save(model.state_dict(), save_path)
        
        #-- script save 
        RS_utils.save_current_file(EXEC_VER)

#---- main 
#train(fabric_, dataloader_, model_, optimizer_)
