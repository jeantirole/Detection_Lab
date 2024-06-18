import torch
from PIL import Image

#--- dataset class 
class Map_Dataset(torch.utils.data.Dataset):      
    def __init__(self, list_IDs,transform,train_path): 
        self.list_IDs = list_IDs
        self.transform = transform
        self.train_path = train_path
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index): 
        ID = self.list_IDs[index] 
        X1 = Image.open(self.train_path + ID + '/street.jpg').convert('RGB')
        #X1 = cv2.resize(X1, (256, 256))
        #X1 = np.asarray(X1)
        #X1 = np.transpose(X1,[1, 2, 0])
        X1 = self.transform(X1)
        #X1 = cv2.resize(X1, (256, 256)) 
        
        #X2 = cv2.imread(train_path + ID + '/orthophoto.tif')  
        #X2 = cv2.resize(X2, (256, 256)) 
        
        #X3 = rasterio.open(train_path + ID + '/s2_l2a.tif').read() 
        #X3 = np.transpose(X3, [1, 2, 0]) 
        
        y = int(open(self.train_path + ID + '/label.txt', "r").read())
        return X1, y 
    
#--

import torch
from PIL import Image
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import timm
import argparse
import yaml
import sys

sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Libs")
import RS_dataset
import RS_models
import RS_utils

import rasterio
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms.functional as F

sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Libs")
import RS_dataset
import RS_models
import RS_utils



#--- dataset class 
class Map_Dataset_v1(torch.utils.data.Dataset):      
    def __init__(self, list_IDs,train_path,max_size): 
        self.list_IDs = list_IDs
        self.train_path = train_path
        self.max_value = max_size
        #self.max_value = 376
        self.min_value = 256
    def __len__(self):
        return len(self.list_IDs)
    
    
    def resize(self,img_):
        resize_transform = torchvision.transforms.Resize(
            size=self.min_value, max_size= self.max_value,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR 
        )
        img_ = resize_transform(img_)
        return img_
    
    
    def centercrop_normalize(self,img_):
        torchvision_transform = torchvision.transforms.Compose([
        
        torchvision.transforms.CenterCrop(self.max_value),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))           
        ])
        
        return torchvision_transform(img_)
    
    
    def ratio_pad(self,img_da):
        # PIL_image = Image.fromarray(bb)

        #--
        max_value = self.max_value
        #width_, height_ = img_da.shape[0],img_da.shape[1]
        width_, height_ = img_da.size
        # left top right bottom 
        img_da = F.pad(img=img_da,padding=[(max_value-width_)//2+1, (max_value-height_)//2+1, (max_value-width_)//2+1,(max_value-height_)//2+1], padding_mode="constant", fill=0)    
        #img_da = F.pad(img=img_da,padding=[0, max_value-height_, max_value-width_,0], padding_mode="constant", fill=0)
        
        return img_da
    
    
    def __getitem__(self, index): 
        ID = self.list_IDs[index] 
        X1 = Image.open(self.train_path + ID + '/street.jpg').convert('RGB')
        #X1 = cv2.resize(X1, (256, 256))
        #X1 = np.asarray(X1)
        #X1 = np.transpose(X1,[1, 2, 0])
        
        X1 = self.resize(X1)
        X1 = self.ratio_pad(X1)
        X1 = self.centercrop_normalize(X1)
        #X1 = cv2.resize(X1, (256, 256)) 
        
        #X2 = Image.open(self.train_path + ID + '/orthophoto.tif')  
        #if self.transform:
        #    X2 = self.transform(X2)
        #X2 = cv2.resize(X2, (256, 256)) 
        
        
        #X3 = rasterio.open(self.train_path + ID + '/s2_l2a.tif').read() 
        #X3 = np.transpose(X3, [1, 2, 0]) 
        #X3 = self.transform(X3)
        
        y = int(open(self.train_path + ID + '/label.txt', "r").read())
        return X1, y 
    
#---