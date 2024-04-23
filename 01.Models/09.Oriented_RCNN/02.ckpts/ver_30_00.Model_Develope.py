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
import time
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
import torch

# VOC class names

#---------------------------------
#-- args operation
EXEC_VER = 30 # zero means test 
DDP = False
TASK = "FACE_Detection"
IMAGES_PATH =  '/mnt/hdd/eric/.tmp_ipy/00.Data/FaceMaskDetection/images/'
IMAGES_PATH_TEST =  '/mnt/hdd/eric/.tmp_ipy/00.Data/FaceMaskDetection/test_images/'
ANNO_PATH = "/mnt/hdd/eric/.tmp_ipy/00.Data/voc/VOCdevkit/VOC2007/Annotations"

#-- args data
IMAGE_HEIGHT = 400
IMAGE_WIDTH  = 400
IMAGE_SIZE = 400
#-- args modeling 
MODEL_NAME = "FasterRCNN"
BATCH_SIZE = 4 
LEARNING_RATE = 0.005 
N_CLASSES = 4
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
    "background", "person", "mask1", "mask2"
]

#-- logger 
log_path = f'/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/09.Oriented_RCNN/01.log/ver_{EXEC_VER}_%Y-%m-%d_%H-%M-%S.log'
logger = RS_utils.log_creator(log_path)

data_transform = transforms.Compose([  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

scale = 1 
data_aug = A.Compose([
    A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
    # min_size보다 작으면 pad
    A.PadIfNeeded(min_height=int(IMAGE_SIZE * scale), min_width=int(IMAGE_SIZE * scale), border_mode=cv2.BORDER_CONSTANT)],
    #A.CenterCrop(height=120, width=120)],
    #A.LongestMaxSize(max_size=200, interpolation=cv2.INTER_LINEAR,always_apply=True)],
    #A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH,interpolation=cv2.INTER_LINEAR)],
    bbox_params=A.BboxParams(format='pascal_voc',label_fields=['category_ids']) )

train_dataset = RS_dataset.MaskDataset_v2(data_transform, IMAGES_PATH, data_aug)
#test_dataset = RS_dataset.MaskDataset(data_transform, IMAGES_PATH_TEST)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.collate_fn, shuffle=DATA_SHUFFLE)
#test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=test_dataset.collate_fn,shuffle=False)

#-------------------------------------------------------------------------------------------

# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# def get_model_instance_segmentation(num_classes):
  
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model


batch_data_all = next(iter(train_dataloader))
sample_batch_img = batch_data_all[0]

#-----------------------------------
# model to calculate the dimension 
model = torchvision.models.resnet50(pretrained=True)
req_layers = list(model.children())[:8]
backbone = nn.Sequential(*req_layers)

# unfreeze all the parameters
for param in backbone.named_parameters():
    param[1].requires_grad = True
    
# run the image through the backbone
out = backbone(sample_batch_img)
OUT_C, OUT_H, OUT_W = out.size(dim=1), out.size(dim=2), out.size(dim=3)

# check the scale 
width_scale_factor = IMAGE_WIDTH // OUT_W
height_scale_factor = IMAGE_HEIGHT // OUT_H

IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
OUT_SIZE = (OUT_H, OUT_W)
ROI_SIZE = (2, 2)

#-- logger 
log_path = f'/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/09.Oriented_RCNN/01.log/ver_{EXEC_VER}_%Y-%m-%d_%H-%M-%S.log'
logger = RS_utils.log_creator(log_path)


#--- model & optimizer 
model = RS_models.TwoStageDetector(IMG_SIZE, OUT_SIZE, OUT_C, N_CLASSES, ROI_SIZE, device=DEVICE)

#model = get_model_instance_segmentation(N_CLASSES)

#model.to(DEVICE)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE,
                                momentum=0.9, weight_decay=0.0005)


print('----------------------train start--------------------------')
iteration= 0
for epoch in range(EPOCHS):
    model.train()

    
    for img_batch, gt_bboxes_batch, gt_classes_batch in train_dataloader:
        
        #--- DDP 
        if DDP != True:
            img_batch, gt_bboxes_batch, gt_classes_batch = img_batch.to(DEVICE), gt_bboxes_batch.to(DEVICE), gt_classes_batch.to(DEVICE)
                

        model = model.to(DEVICE)
        cls_loss, rpn_loss = model(img_batch,gt_bboxes_batch,gt_classes_batch)
        losses = cls_loss + rpn_loss
        
        #-----

        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
        
        log_dict=   {
            "epoch" : epoch,
            "iteration" : iteration,
            "iteration_by_epoch" : (epoch // len(train_dataloader)) // BATCH_SIZE,
            "total loss" : "{:.5f}".format(losses.item()), 
            "rpn loss" : "{:.5f}".format(rpn_loss.item()),  
            "class loss" : "{:.5f}".format(cls_loss.item()),  
            "batch_size" : int(BATCH_SIZE),
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
