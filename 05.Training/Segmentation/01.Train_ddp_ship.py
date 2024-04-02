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
#---
from lightning.fabric import Fabric
import lightning as L


#-- args
EXEC_VER = 23 
BATCH_SIZE = 32
#DEVICE = "cuda:0"
DEVICES = [0,1,2,3]
RESUME = False
SAVE_EPOCH = 20

#-- data
img_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/01.512_imgs"
mask_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/02.512_masks"

img_path_ship  = np.array(sorted(glob(os.path.join(img_path, "*.png"))) )
mask_path_ship = np.array(sorted(glob(os.path.join(mask_path, "*.png"))) )

aa = np.load("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Training/Segmentation/03.data_list/512_ships.npy")

selected_paths_img = img_path_ship[aa]
selected_paths_mask  = mask_path_ship[aa]

#-- category 
ISAID_CLASSES_SHIP = (
    'background','ship','harbor' 
    )
ISAID_PALETTE_SHIP = {
    0: (0, 0, 0), 
    1: (0, 0, 63), 
    2: (0, 100, 155)}


#-- ddp 
fabric = L.Fabric(accelerator="cuda", devices=DEVICES, strategy="ddp")
fabric.launch()


#--- logger
# Set up logging
log_filename = datetime.datetime.now().strftime(f'./01.log/ver_{EXEC_VER}_%Y-%m-%d_%H-%M-%S.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.FileHandler(log_filename)
logger.addHandler(handler)

#-- datasets 
train_dataset = dataset_rs.Seg_RS_dataset_ship(img_dir=selected_paths_img, mask_dir=selected_paths_mask, image_resize = None, phase="train",palette=ISAID_PALETTE_SHIP )
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn)
valid_dataset = dataset_rs.Seg_RS_dataset_ship(img_dir=selected_paths_img, mask_dir=selected_paths_mask, image_resize = None, phase="train",palette=ISAID_PALETTE_SHIP )
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=valid_dataset.collate_fn)

data_loader= {}
data_loader["train"] = train_loader
data_loader["valid"] = valid_loader

#-- model 
model = smp.UnetPlusPlus(
    encoder_name="resnet152",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)

#-- resume
if RESUME == True:
    tgt_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Training/Segmentation/02.ckpts"
    ckpt_path = os.path.join( tgt_path, sorted(os.listdir(tgt_path))[-1]  )
    print("resume chekcpoint : ",ckpt_path)

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)

#model = model.to(DEVICE)

#-- loss 
#criterion = nn.CrossEntropyLoss()
criterion = dataset_rs.UNet_metric(num_classes=3)

#-- optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# run
epochs = 999
iteration = 0
total_loss = 0
model, optimizer = fabric.setup(model, optimizer)
dataloader = data_loader["train"]
dataloader = fabric.setup_dataloaders(dataloader)

for epoch in range(epochs):
    
    iteration = 0
    epoch_running_loss = 0 
    
    #tqdm_iterator = tqdm(data_loader["train"], desc=f"Epoch {epoch}")
    
    for index, data in enumerate(dataloader):
        
        imgs, masks = data
        #imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        
        # opt
        optimizer.zero_grad()
        
        # runs
        outputs = model(imgs)
        
        # loss
        loss,dice_score = criterion(outputs, masks)
        #loss.backward()
        fabric.backward(loss)
        optimizer.step()
        
        # stat
        epoch_running_loss += loss.item()

        # log
        logger.info(f"epoch : {epoch} iter : {index} loss: {loss:.5f} dice: {dice_score:.5f}" )
        log = {'loss': f'{loss / 10:.5f}' }
        #wandb.log(log)

        log_iter = 100
        
        if (index % log_iter) == 0:    # print every 2000 mini-batches
            print(f"epoch : {epoch} , iter : {index} , total_iter : {len(data_loader['train'])} , running_loss : {epoch_running_loss / (index +1)}")
        
    
    #-- save 
    
    if epoch % SAVE_EPOCH ==0:
        #-- save 
        save_path = f"./02.ckpts/ver_{EXEC_VER}_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), save_path)