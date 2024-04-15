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

images_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/voc/VOCdevkit/VOC2007/JPEGImages"
anno_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/voc/VOCdevkit/VOC2007/Annotations"

# VOC class names
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

image_size_ = 600
dataset_ = RS_dataset.Detection_Dataset_v1(images_path, anno_path, classes, image_size=image_size_)
loader_ = DataLoader(dataset_, batch_size=8, collate_fn=dataset_.collate_fn)

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

out_c, out_h, out_w = out.size(dim=1), out.size(dim=2), out.size(dim=3)

# check the scale 

width_scale_factor = image_size_ // out_w
height_scale_factor = image_size_ // out_h


img_height = 600 
img_width  = 600
img_size = (img_height, img_width)
out_size = (out_h, out_w)
n_classes = 20 # exclude pad idx
roi_size = (2, 2)
DEVICE = "cuda:0"

detector = RS_models.TwoStageDetector(img_size, out_size, out_c, n_classes, roi_size, device=DEVICE)

# detector.eval()
# total_loss = detector(img_batch, gt_bboxes_batch, gt_classes_batch)
# proposals_final, conf_scores_final, classes_final = detector.inference(img_batch)


learning_rate = 1e-3
n_epochs = 1

#from tqdm import tqdm 


detector.train()
detector.to(DEVICE)
optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)

loss_list = []
iteration= 0
for i in range(n_epochs):
    total_loss = 0
    
    for img_batch, gt_bboxes_batch, gt_classes_batch in loader_:
        
        img_batch, gt_bboxes_batch, gt_classes_batch = img_batch.to(DEVICE), gt_bboxes_batch.to(DEVICE), gt_classes_batch.to(DEVICE)
        # forward pass
        iteration += 1 
        
        #print(img_batch.shape)
        loss = detector(img_batch, gt_bboxes_batch, gt_classes_batch)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if iteration % 10 == 0:
            print("iteration : ",iteration, f"/ {len(loader_)}")
            print(loss.item())


    loss_list.append(total_loss)
            

#loss_list = training_loop(detector, learning_rate, loader_, n_epochs)

#plt.plot(loss_list)

torch.save(detector.state_dict(), "model.pt")

# inference 
detector.eval()
proposals_final, conf_scores_final, classes_final = detector.inference(img_batch, conf_thresh=0.99, nms_thresh=0.05)

# project proposals to the image space
prop_proj_1 = project_bboxes(proposals_final[0], width_scale_factor, height_scale_factor, mode='a2p')
prop_proj_2 = project_bboxes(proposals_final[1], width_scale_factor, height_scale_factor, mode='a2p')

# get classes
classes_pred_1 = [idx2name[cls] for cls in classes_final[0].tolist()]
classes_pred_2 = [idx2name[cls] for cls in classes_final[1].tolist()]

nrows, ncols = (1, 2)
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

fig, axes = display_img(img_batch, fig, axes)
fig, _ = display_bbox(prop_proj_1, fig, axes[0], classes=classes_pred_1)
fig, _ = display_bbox(prop_proj_2, fig, axes[1], classes=classes_pred_2)
