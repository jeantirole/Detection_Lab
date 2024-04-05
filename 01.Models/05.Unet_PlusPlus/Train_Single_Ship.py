import RS_utils
import RS_dataset
import RS_models
#---
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
MODEL_NAME = "UNET_PP"
EXEC_VER = 35 
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
log_filename = datetime.datetime.now().strftime(f'./01.log/ver_{EXEC_VER}_%Y-%m-%d_%H-%M-%S.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.FileHandler(log_filename)
logger.addHandler(handler)

#-- dataset
train_dataset = RS_dataset.Seg_RS_dataset_ship(img_dir=selected_paths_img, mask_dir=selected_paths_mask, image_resize = None, phase="train",palette=ISAID_PALETTE_SHIP )
dataloader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn)

#--- model 
#model = RS_models.Edge_Net()
model = smp.UnetPlusPlus(encoder_name="resnet152",classes=3)
model = model.to(DEVICE)
#criterion = nn.CrossEntropyLoss(reduction="mean") 
#criterion = nn.BCELoss()
criterion = RS_dataset.UNet_metric(num_classes=3)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

#-- resume
if RESUME == True:
    tgt_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Training/Segmentation/02.ckpts"
    ckpt_path = os.path.join( tgt_path, sorted(os.listdir(tgt_path))[-1]  )
    print("resume chekcpoint : ",ckpt_path)

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)

#--- fabric setup 



epochs = 999
iteration = 0
for epoch in range(epochs):
    
    
    epoch_running_loss = 0 
    
    #tqdm_iterator = tqdm(data_loader["train"], desc=f"Epoch {epoch}")
    
    for index, data in enumerate(dataloader):
        
        imgs, masks = data
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
    
        # opt
        optimizer.zero_grad()
        
        # runs
        outputs = model(imgs)
        
        # loss
        loss,dice_score = criterion(outputs, masks)
        loss.backward()
        #fabric.backward(loss)
        optimizer.step()
        
        # stat
        epoch_running_loss += loss.item()

        
        # log
        dice_score = 0
        logger.info(f"epoch : {epoch} iter : {index} loss: {loss:.5f} dice: {dice_score:.5f}" )
        log = {'loss': f'{loss / 10:.5f}' }
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