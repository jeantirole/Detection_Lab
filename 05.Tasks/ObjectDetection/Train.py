import sys 
sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Libs")
import RS_dataset
import RS_models
import RS_utils

import torchvision
import torch.nn as nn 
from torchvision.datasets import VOCDetection
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor, to_pil_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import torch 
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import collections
#---
from lightning.fabric import Fabric
import lightning as L


#-- data 
images_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/voc/VOCdevkit/VOC2007/JPEGImages"
anno_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/voc/VOCdevkit/VOC2007/Annotations"

# VOC class names

#---------------------------------
#-- args operation
EXEC_VER = 8 # zero means test 
DDP = False
TASK = "Object_Detection"
#-- args data
IMAGE_SIZE = 600
#-- args modeling 
MODEL_NAME = "FasterRCNN"
BATCH_SIZE = 8 # 4
LEARNING_RATE = 1e-4 #1e-5
N_CLASSES = 20
#-- args train
DEVICE = "cuda:0"
DEVICES = [0,1,2,3]
RESUME = False
SAVE_EPOCH = 10
EPOCHS = 140
DATA_SHUFFLE = True
#-- args category
# VOC class names 
CALSSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

dataset_ = RS_dataset.Detection_Dataset_v2(images_path, anno_path, CALSSES, image_size=IMAGE_SIZE,normalizing=True)
train_dataloader = DataLoader(dataset_, batch_size=BATCH_SIZE, collate_fn=dataset_.collate_fn,shuffle=DATA_SHUFFLE)

#--------------------------------
# grab a batch for demonstration

for img_batch, gt_bboxes_batch, gt_classes_batch in train_dataloader:
    img_data_all = img_batch
    gt_bboxes_all = gt_bboxes_batch
    gt_classes_all = gt_classes_batch
    break


#-----------------------------------
# model to calculate the dimension 
model = torchvision.models.resnet50(pretrained=True)
req_layers = list(model.children())[:8]
backbone = nn.Sequential(*req_layers)

# unfreeze all the parameters
for param in backbone.named_parameters():
    param[1].requires_grad = True
    
# run the image through the backbone
out = backbone(img_data_all)
OUT_C, OUT_H, OUT_W = out.size(dim=1), out.size(dim=2), out.size(dim=3)

# check the scale 
width_scale_factor = IMAGE_SIZE // OUT_W
height_scale_factor = IMAGE_SIZE // OUT_H

IMG_HEIGHT = 600 
IMG_WIDTH  = 600
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
OUT_SIZE = (OUT_H, OUT_W)
ROI_SIZE = (2, 2)

#-- logger 
log_path = f'/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Tasks/ObjectDetection/01.log/ver_{EXEC_VER}_%Y-%m-%d_%H-%M-%S.log'
logger = RS_utils.log_creator(log_path)


#--- model & optimizer 
model = RS_models.TwoStageDetector(IMG_SIZE, OUT_SIZE, OUT_C, N_CLASSES, ROI_SIZE, device=DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

if DDP:
    #--- fabric setup 
    fabric = L.Fabric(accelerator="cuda", devices=DEVICES, strategy="ddp")
    fabric.launch()
    
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(train_dataloader)
else:
    model = model.to(DEVICE)

#--- fabric setup 


iteration= 0
for epoch in range(EPOCHS):
    
    epoch_running_loss = 0
    
    for img_batch, gt_bboxes_batch, gt_classes_batch in train_dataloader:
        
        #--- DDP 
        if DDP != True:
            img_batch, gt_bboxes_batch, gt_classes_batch = img_batch.to(DEVICE), gt_bboxes_batch.to(DEVICE), gt_classes_batch.to(DEVICE)
        
        # opt 
        optimizer.zero_grad()

        # runs 
        loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)
        
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
            #"dice" : "{:.5f}".format(coef_dice.item()),
            "lr" : optimizer.param_groups[0]['lr']
        }        
        logger.info(log_dict)
        iteration += 1
    
    #-- save 
    if epoch % SAVE_EPOCH ==0:
    #-- weight save 
        save_path = f"./02.ckpts/ver_{EXEC_VER}_{TASK}_{MODEL_NAME}_epoch_{epoch}_iteration_{iteration}.pt"
        torch.save(model.state_dict(), save_path)
        
        #-- script save 
        RS_utils.save_current_file(EXEC_VER)
