
import sys
sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Libs")
import RS_dataset
import RS_models
import RS_utils
import RS_train

import datetime
import logging
import numpy as np 
from glob import glob
import os
import torch
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
#---
import torch.nn as nn 
#---
from lightning.fabric import Fabric
import lightning as L
import segmentation_models_pytorch as smp

#-- data
img_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/01.512_imgs"
mask_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/02.512_masks"

img_path_ship  = np.array(sorted(glob(os.path.join(img_path, "*.png"))) )
mask_path_ship = np.array(sorted(glob(os.path.join(mask_path, "*.png"))) )

aa = np.load("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Training/Segmentation/03.data_list/512_ships.npy")

selected_paths_img = img_path_ship[aa]
selected_paths_mask  = mask_path_ship[aa]


#-- args
TASK = "SHIP"
MODEL_NAME = "UNET_with_EDGE_v"
EXEC_VER = 70 
BATCH_SIZE = 4
DEVICE = "cuda:0"
DEVICES = [0,1,2,3]
RESUME = False
SAVE_EPOCH = 20

#-- category 
ISAID_CLASSES_SHIP = (
    'background','ship','harbor' 
    )
ISAID_PALETTE_SHIP = {
    0: (0, 0, 0), 
    1: (0, 0, 63), 
    2: (0, 100, 155)}


#--- logger
# Set up logging
log_filename = datetime.datetime.now().strftime(f'/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/08.Segmentation_with_Edge/01.log/ver_{EXEC_VER}_%Y-%m-%d_%H-%M-%S.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.FileHandler(log_filename)
logger.addHandler(handler)


#-- dataset
train_dataset = RS_dataset.Seg_RS_dataset_edge_v2(img_dir=selected_paths_img, mask_dir=selected_paths_mask, 
                                                  image_resize = None, phase="train",palette=ISAID_PALETTE_SHIP,
                                                  gaussian=True,
                                                  mask_onehot=True, 
                                                  sigmoid=True)
dataloader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn)

#--- model 
#model = RS_models.Edge_Net()
model_1 = smp.UnetPlusPlus(encoder_name="resnet152",classes=3)
tgt_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/07.Unet_PlusPlus_EdgeNet/02.ckpts/ver_35_unet_epoch_101_iteration_470256.pt"
checkpoint = torch.load(tgt_path)
model_1.load_state_dict(checkpoint)


model_2 = RS_models.Edge_Net(input_channel=3)
tgt_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/06.Edge_Net/02.ckpts/ver_62_EDGE_EDGE_NET_epoch_81_iteration_6081318.pt"
checkpoint = torch.load(tgt_path)
model_2.load_state_dict(checkpoint)


criterion = RS_dataset.UNet_metric(num_classes=3)

model = RS_models.CombinedModel(model_1, model_2)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


epochs = 999
iteration = 0
for epoch in range(epochs):
    epoch_running_loss = 0 

    dice_score = 0
    for index, data in enumerate(dataloader):
        imgs, original_mask, encoded_mask = data
        imgs, original_mask, encoded_mask = imgs.to(DEVICE), original_mask.to(DEVICE), encoded_mask.to(DEVICE)
    
        # opt
        optimizer.zero_grad()
        # runs
        outputs,loss_percept = model(imgs, encoded_mask)
        # loss
        loss_seg,dice_score = criterion(outputs, original_mask)
        
        loss = loss_seg + loss_percept
        loss.backward()
        #fabric.backward(loss)
        optimizer.step()

        # stat
        epoch_running_loss += loss.item()

        # log
        logger.info(f"epoch : {epoch} iter : {index} loss: {loss:.5f} dice: {dice_score:.5f} loss_percept: {loss_percept:.5f}" )
        log = {'loss': f'{loss:.5f}' }
        print(log)
        #wandb.log(log)

        iteration += index
        log_iter = 100
        
        if (index % log_iter) == 0:    # print every log_iter mini-batches
            print(f"epoch : {epoch} , total_iter : {iteration} , iter_per_epoch : {len(dataloader)} , running_loss : {epoch_running_loss / (index +1)}")
        
    #-- save     
    if epoch % SAVE_EPOCH ==0:
        #-- save 
        save_path = f"./02.ckpts/ver_{EXEC_VER}_{TASK}_{MODEL_NAME}_epoch_{epoch + 1}_iteration_{iteration}.pt"
        torch.save(model.state_dict(), save_path)