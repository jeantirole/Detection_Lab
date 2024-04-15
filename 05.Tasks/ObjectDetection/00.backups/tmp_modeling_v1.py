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

#-- data 
images_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/voc/VOCdevkit/VOC2007/JPEGImages"
anno_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/voc/VOCdevkit/VOC2007/Annotations"

# VOC class names

#---------------------------------
#-- args operation
EXEC_VER = 5 # zero means test 
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

dataset_ = RS_dataset.Detection_Dataset_v1(images_path, anno_path, CALSSES, image_size=IMAGE_SIZE)
loader_ = DataLoader(dataset_, batch_size=BATCH_SIZE, collate_fn=dataset_.collate_fn,shuffle=DATA_SHUFFLE)

#--------------------------------
# grab a batch for demonstration

for img_batch, gt_bboxes_batch, gt_classes_batch in loader_:
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
# n_classes = 20 # exclude pad idx
ROI_SIZE = (2, 2)
DEVICE = "cuda:0"

model = RS_models.TwoStageDetector(IMG_SIZE, OUT_SIZE, OUT_C, N_CLASSES, ROI_SIZE, device=DEVICE)


from tqdm import tqdm 


model.train()
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_list = []
iteration= 0
for i in range(EPOCHS):
    total_loss = 0
    
    for img_batch, gt_bboxes_batch, gt_classes_batch in loader_:
        
        img_batch, gt_bboxes_batch, gt_classes_batch = img_batch.to(DEVICE), gt_bboxes_batch.to(DEVICE), gt_classes_batch.to(DEVICE)
        # forward pass
        iteration += 1 
        
        #print(img_batch.shape)
        loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if iteration % 10 == 0:
            print("iteration : ",iteration, f"/ {len(loader_)}")
            print(loss.item())


    loss_list.append(total_loss)
            

#loss_list = training_loop(model, learning_rate, loader_, n_epochs)

#plt.plot(loss_list)

#torch.save(model.state_dict(), "model.pt")

# inference 
saved_model_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Tasks/ObjectDetection/02.ckpts/model.pt"
model.load_state_dict(torch.load(saved_model_path))
model.eval()

proposals_final, conf_scores_final, classes_final = model.inference(img_batch, conf_thresh=0.99, nms_thresh=0.05)

# project proposals to the image space
prop_proj_1 = RS_utils.project_bboxes(proposals_final[0], width_scale_factor, height_scale_factor, mode='a2p')
prop_proj_2 = RS_utils.project_bboxes(proposals_final[1], width_scale_factor, height_scale_factor, mode='a2p')



# get classes
name2idx = {'pad': -1, 'camel': 0, 'bird': 1}
classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

idx2name = {}
for i, name in enumerate(classes):
    idx2name[name] = i
idx2name['pad'] = -1 

classes_pred_1 = [idx2name[cls] for cls in classes_final[0].tolist()]
classes_pred_2 = [idx2name[cls] for cls in classes_final[1].tolist()]



# nrows, ncols = (1, 2)
# fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

# fig, axes = display_img(img_batch, fig, axes)
# fig, _ = display_bbox(prop_proj_1, fig, axes[0], classes=classes_pred_1)
# fig, _ = display_bbox(prop_proj_2, fig, axes[1], classes=classes_pred_2)
