
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from glob import glob
import os

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from tensorflow.keras.utils import to_categorical

#---------------------------------------
import custom_dataset
import utils_rs

#----------------------------------------------
import swin
import upper_net_mmseg
import models


# model swin
model_swin = swin.swin()

# model upernet
model_upernet = upper_net_mmseg.UPerHead(
    in_channels = model_swin.out_channels[1:],
    channels = model_swin.out_channels[2],
    in_index= (0,1,2,3),
    dropout_ratio=0.1,
    norm_cfg= dict(type='SyncBN', requires_grad=True)
)


# model samrs 
swin_samrs = models.SamRS(model1=model_swin, model2=model_upernet)

w_path = "./swint_upernet_imp_sep_model.pth"
weights_ = torch.load(w_path, map_location=torch.device('cpu'))
swin_samrs.load_state_dict(  weights_['state_dict'] )


# model fine-tune initialize 
n_classes = 16
swin_samrs.semseghead_1 = nn.Sequential(
                                    nn.Dropout2d(0.1),
                                    nn.Conv2d(model_swin.out_channels[2], n_classes, kernel_size=1)
                                    )


# class map 
ISAID_CLASSES = ('background', 'ship', 'store_tank', 'baseball_diamond',
               'tennis_court', 'basketball_court', 'Ground_Track_Field',
               'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
               'Harbor')

ISAID_PALETTE = {
    0: (0, 0, 0), 1: (0, 0, 63), 2: (0, 63, 63), 3: (0, 63, 0), 4: (0, 63, 127),
    5: (0, 63, 191), 6: (0, 63, 255), 7: (0, 127, 63), 8: (0, 127, 127),
    9: (0, 0, 127), 10: (0, 0, 191), 11: (0, 0, 255), 12: (0, 191, 127),
    13: (0, 127, 191), 14: (0, 127, 255), 15: (0, 100, 155)}


img_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/04.1024_imgs"
mask_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/08.1024_masks_categorized_imgs_all"

assert len(os.listdir(img_path)) == len(os.listdir(mask_path))

segdataset = custom_dataset.SegDataset(img_path, mask_path, 512)


#-----------------------------------------

import wandb
import logging
from tqdm import tqdm
from lightning.fabric import Fabric
import lightning as L 
import loss_

# distribute gpus
device_list = [0,1,2,3]
fabric = L.Fabric(accelerator="cuda", devices=device_list, strategy='ddp')
fabric.launch()

# log 
logging.basicConfig(filename='./1.log/model_v1.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

wandb.init( project="samrs_swn_v1" )


# opt 
optimizer = torch.optim.AdamW(swin_samrs.parameters(), lr=1e-5)

# model 
# swin_samrs = swin_samrs.train()

# fabric model optimizer
swin_samrs, optimizer =  fabric.setup(swin_samrs, optimizer)


# dataset 
batch_size= 4 
segdataset = custom_dataset.SegDataset(img_path, mask_path, 512)
dataloader = torch.utils.data.DataLoader(segdataset, batch_size=batch_size, shuffle=True)
dataloader = fabric.setup_dataloaders(dataloader)

# Define loss
criterion = nn.CrossEntropyLoss()
criterion = loss_.DiceLoss()

# run 
epochs = 999
for epoch in range(epochs):

    iteration = 0
    running_loss = 0 
    
    for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):

        img, mask = data

        # opt 
        optimizer.zero_grad()
        
        # run
        outputs = swin_samrs(img)
        
        # criterion
        loss = criterion.calculate_average_dice_loss(outputs, mask)
        #print(loss)
        
        #Loss.backward()
        fabric.backward(loss)
        optimizer.step()

        # stat
        running_loss += loss.item()

        # log
        logger.info(f"[{epoch}, {i}] loss: {loss:.8f}")

        
        log = {'loss': f'{loss / 10:.8f}' }
        #print(log)
        wandb.log(log)

        log_iter = 200
        if (i % log_iter) == 0:    # print every 2000 mini-batches
            print(f"epoch : {epoch} iter : {i} /  total_iter : {len(dataloader)} running_loss : {running_loss / log_iter}")
            
            running_loss = 0.0
         
            
        #-----
    #-- epoch
    save_path = f"./2.ckpts/swin_rs_{epoch + 1}.pt"
    torch.save(swin_samrs.state_dict(), save_path)


