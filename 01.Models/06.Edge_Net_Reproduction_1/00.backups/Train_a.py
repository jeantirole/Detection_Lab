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

#-- data
img_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/01.512_imgs"
mask_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/02.512_masks"

img_path_ship  = np.array(sorted(glob(os.path.join(img_path, "*.png"))) )
mask_path_ship = np.array(sorted(glob(os.path.join(mask_path, "*.png"))) )

aa = np.load("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Training/Segmentation/03.data_list/512_ships.npy")

selected_paths_img = img_path_ship[aa]
selected_paths_mask  = mask_path_ship[aa]

#-- args
TASK = "EDGE"
MODEL_NAME = "EDGE_NET"
EXEC_VER = 21 # zero means test 
BATCH_SIZE = 4
DEVICE = "cuda:0"
DEVICES = [0,1,2,3]
RESUME = False
SAVE_EPOCH = 20
EPOCHS = 140
DATA_SHUFFLE = True

#-- category 
ISAID_CLASSES_SHIP = (
    'background','ship','harbor' 
    )
ISAID_PALETTE_SHIP = {
    0: (0, 0, 0), 
    1: (0, 0, 63), 
    2: (0, 100, 155)
    }

#--- logger
# Set up logging
log_filename = datetime.datetime.now().strftime(f'/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/06.Edge_Net_Reproduction_1/01.log/ver_{EXEC_VER}_%Y-%m-%d_%H-%M-%S.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.FileHandler(log_filename)
logger.addHandler(handler)



#-- dataset
train_dataset = RS_dataset.Seg_RS_dataset_edge_v3(img_dir=selected_paths_img, mask_dir=selected_paths_mask, image_resize = None, phase="train",palette=ISAID_PALETTE_SHIP,gaussian=False,mask_onehot=False,softmax=False)
dataloader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=DATA_SHUFFLE, collate_fn=train_dataset.collate_fn)
#--- model 
model = RS_models.Edge_Net(input_channel=len(ISAID_PALETTE_SHIP.keys()), output_channel=2)
model = model.to(DEVICE)
#criterion = nn.CrossEntropyLoss(reduction="mean") 
#criterion = RS_utils.UNet_metric(num_classes= 2)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)



#--- fabric setup 
iteration = 0
for epoch in range(EPOCHS):
    
    epoch_running_loss = 0 
    
    for index, data in enumerate(dataloader):
        
        _, masks, _, edge = data
    
        #---
        masks, edge = masks.to(DEVICE), edge.to(DEVICE)
        masks = masks.float()
        edge = RS_utils.apply_one_hot(edge, n_class=2, batch=True)

        # opt
        optimizer.zero_grad()

        # runs
        outputs = model(masks)
        
        #--- get dice 
        # loss
        loss = criterion(outputs, edge)
        
        coef_dice = RS_utils.dice_coef(outputs, edge, n_class=2)
        
        #loss_ce, loss_dice, coef_dice = criterion(outputs, edge)
        #loss = loss_ce + loss_dice
        loss.backward()
        #fabric.backward(loss)
        optimizer.step()
        
        # stat
        epoch_running_loss += loss.item()

        loss_percept = 0
        
        log_dict=   {
            "epoch" : epoch,
            "iteration" : iteration,
            "loss" : "{:.5f}".format(loss.item()),  # Formatting loss to 5 decimal places
            "dice" : "{:.5f}".format(coef_dice.item())  # Formatting dice coefficient to 5 decimal places
        }        
        logger.info(log_dict)

        iteration += index
        log_iter = 100

    #-- save 
    
    if epoch % SAVE_EPOCH ==0:
        #-- weight save 
        save_path = f"./02.ckpts/ver_{EXEC_VER}_{TASK}_{MODEL_NAME}_epoch_{epoch + 1}_iteration_{iteration}.pt"
        torch.save(model.state_dict(), save_path)
        
        #-- script save 
        RS_utils.save_current_file(EXEC_VER)