import RS_utils
import RS_dataset
import RS_models

import numpy as np 
from glob import glob
import os
import torch
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
#---
import torch.nn as nn 


#-- data
img_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/01.512_imgs"
mask_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/02.512_masks"

img_path_ship  = np.array(sorted(glob(os.path.join(img_path, "*.png"))) )
mask_path_ship = np.array(sorted(glob(os.path.join(mask_path, "*.png"))) )

aa = np.load("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Training/Segmentation/03.data_list/512_ships.npy")

selected_paths_img = img_path_ship[aa]
selected_paths_mask  = mask_path_ship[aa]


#-- args
EXEC_VER = 30 
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


#-- dataset
train_dataset = RS_dataset.Seg_RS_dataset_edge(img_dir=selected_paths_img, mask_dir=selected_paths_mask, image_resize = None, phase="train",palette=ISAID_PALETTE_SHIP )
dataloader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn)

#--- model 
model = RS_models.Edge_Net()
model = model.to(DEVICE)
#criterion = nn.CrossEntropyLoss(reduction="mean") 
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


epochs = 999
for epoch in range(epochs):
    
    iteration = 0
    epoch_running_loss = 0 
    
    #tqdm_iterator = tqdm(data_loader["train"], desc=f"Epoch {epoch}")
    
    for index, data in enumerate(dataloader):
        
        masks, masks_edge = data
    
        #---
        masks, masks_edge = masks.to(DEVICE), masks_edge.float().to(DEVICE)

        # opt
        optimizer.zero_grad()
        
        # runs
        outputs = model(masks)
        
        # criterion 
        # this is so important for BCELoss ! 
        
        # this is for Cross-Entropy
        #outputs = outputs.squeeze(1)       
        #masks_edge = masks_edge.squeeze(1)
        
        # this is for Binary-Cross-Entropy
        outputs = nn.functional.sigmoid(outputs)

        #print(outputs.shape)
        #print(masks_edge.shape)

        # loss
        loss = criterion(outputs, masks_edge)
        loss.backward()
        #fabric.backward(loss)
        optimizer.step()
        
        # stat
        epoch_running_loss += loss.item()

        # log
        #logger.info(f"epoch : {epoch} iter : {index} loss: {loss:.5f} dice: {dice_score:.5f}" )
        log = {'loss': f'{loss / 10:.5f}' }
        print(log)
        #wandb.log(log)

        log_iter = 100
        
        if (index % log_iter) == 0:    # print every 2000 mini-batches
            print(f"epoch : {epoch} , iter : {index} , total_iter : {len(dataloader)} , running_loss : {epoch_running_loss / (index +1)}")
        
    
    #-- save 
    
    if epoch % SAVE_EPOCH ==0:
        #-- save 
        save_path = f"./02.ckpts/ver_{EXEC_VER}_edgenet_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), save_path)