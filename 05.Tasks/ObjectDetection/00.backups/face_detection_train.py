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


#------------------------------------------------------------------------------------------------

# data_transform = transforms.Compose([  # transforms.Compose : list 내의 작업을 연달아 할 수 있게 호출하는 클래스
#         transforms.ToTensor() # ToTensor : numpy 이미지에서 torch 이미지로 변경
#     ])

# def collate_fn(batch):
#     return tuple(zip(*batch))


# dataset = RS_dataset.MaskDataset(data_transform, '/mnt/hdd/eric/.tmp_ipy/00.Data/FaceMaskDetection/images')
# #test_dataset = MaskDataset(data_transform, 'test_images/')

# data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
# #test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)

#-- data 
images_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/voc/VOCdevkit/VOC2007/JPEGImages"
anno_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/voc/VOCdevkit/VOC2007/Annotations"

# VOC class names

#---------------------------------
#-- args operation
EXEC_VER = 22 # zero means test 
DDP = False
TASK = "FACE_Detection"
#-- args data
IMAGE_SIZE = 600
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
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

#-- logger 
log_path = f'/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Tasks/ObjectDetection/01.log/ver_{EXEC_VER}_%Y-%m-%d_%H-%M-%S.log'
logger = RS_utils.log_creator(log_path)

data_transform = transforms.Compose([  # transforms.Compose : list 내의 작업을 연달아 할 수 있게 호출하는 클래스
        transforms.ToTensor() # ToTensor : numpy 이미지에서 torch 이미지로 변경
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

dataset_ = RS_dataset.MaskDataset(data_transform, '/mnt/hdd/eric/.tmp_ipy/00.Data/FaceMaskDetection/images/')
#test_dataset = RS_dataset.MaskDataset(data_transform, 'test_images/')

train_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=BATCH_SIZE, collate_fn=collate_fn)
#test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn,shuffle=DATA_SHUFFLE)

#-------------------------------------------------------------------------------------------

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_instance_segmentation(num_classes):
  
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model



model = get_model_instance_segmentation(4)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE,
                                momentum=0.9, weight_decay=0.0005)


print('----------------------train start--------------------------')
iteration= 0
for epoch in range(EPOCHS):
    model.train()

    
    for img_batch, annotations in train_dataloader:
        
        #--- DDP 
        # if DDP != True:
        #     img_batch, gt_bboxes_batch, gt_classes_bathc = img_batch.to(DEVICE), gt_bboxes_batch.to(DEVICE), gt_classes_batch.to(DEVICE)
        
        # annotations = {
        #     'bbox':gt_bboxes_batch,
        #     'label':gt_classes_batch
        # }
        #imgs = list(img.to(device) for img in imgs)
        imgs = list(img.to(device) for img in img_batch)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations) 
        
        #-----
        losses = sum(loss for loss in loss_dict.values())        

        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
        
        log_dict=   {
            "epoch" : epoch,
            "iteration" : iteration,
            "loss" : "{:.5f}".format(losses.item()),  
            "batch_size" : "{:.5f}".format(BATCH_SIZE),
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
