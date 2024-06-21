import torch
from PIL import Image
import albumentations as A 

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


#--- dataset class 
class Map_Dataset_v2(torch.utils.data.Dataset):      
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
            
            #--- v2 version 
            #==> bilinear => bicubic 
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC 
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




#--- dataset class 
#
# v2 => v3 
# cfg.INTERPOLATION Option added 
#
#---
class Map_Dataset_v3(torch.utils.data.Dataset):      
    def __init__(self, list_IDs,train_path, max_size, cfg): 
        self.list_IDs = list_IDs
        self.train_path = train_path
        self.max_value = max_size
        self.min_value = int(self.max_value* 2/3) # min : max = 2 :3
        self.cfg = cfg 
    
    def __len__(self):
        return len(self.list_IDs)
    
    
    def resize(self,img_):
        if self.cfg.INTERPOLATION == "bilinear":
            inter_ = torchvision.transforms.InterpolationMode.BICUBIC
        elif self.cfg.INTERPOLATION == "bicubic":
            inter_ = torchvision.transforms.InterpolationMode.BILINEAR
        
        resize_transform = torchvision.transforms.Resize(
            size=self.min_value, max_size= self.max_value,
             interpolation=inter_ 
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
    
    
#--- dataset class
# v3 => v4 
# augmentation added
#--- 
class Map_Dataset_v4(torch.utils.data.Dataset):      
    def __init__(self, list_IDs,train_path, max_size, cfg): 
        self.list_IDs = list_IDs
        self.train_path = train_path
        self.max_value = max_size
        self.min_value = int(self.max_value* 2/3) # min : max = 2 :3
        self.cfg = cfg
         
    
    def __len__(self):
        return len(self.list_IDs)
    
    
    def resize(self,img_):
        if self.cfg.INTERPOLATION == "bilinear":
            inter_ = torchvision.transforms.InterpolationMode.BICUBIC
        elif self.cfg.INTERPOLATION == "bicubic":
            inter_ = torchvision.transforms.InterpolationMode.BILINEAR
        
        resize_transform = torchvision.transforms.Resize(
            size=self.min_value, max_size= self.max_value,
             interpolation=inter_ 
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
        
        max_value = self.max_value       
        width_, height_ = img_da.size
        # left top right bottom 
        img_da = F.pad(img=img_da,padding=[(max_value-width_)//2+1, (max_value-height_)//2+1, (max_value-width_)//2+1,(max_value-height_)//2+1], padding_mode="constant", fill=0)    
        #img_da = F.pad(img=img_da,padding=[0, max_value-height_, max_value-width_,0], padding_mode="constant", fill=0)
        
        return img_da
    
    
    def augmentations(self,img):
        
        prob = 0.5
        AUGMENTATIONS = {
            "griddropout": lambda prob: A.OneOf([A.GridDropout(p=prob, holes_number_x=3, holes_number_y=4), A.GridDropout(p=prob)], p=prob),
            "horizontalflip": lambda prob: A.HorizontalFlip(p=prob),
            "gaussnoise": lambda prob: A.GaussNoise(p=prob)
        }
        
        img = np.asarray(img)
    

        config = ["griddropout", "horizontalflip", "gaussnoise"]
        # Create the list of transformations based on the configuration
        transforms_list = [AUGMENTATIONS[aug](prob) for aug in self.cfg.AUGMENTATIONS]

        # Compose the transformations
        albumentations_transforms = A.Compose(transforms_list)
        
        # albumentations_transforms = A.Compose([
        #     A.OneOf([A.GridDropout(p=prob, holes_number_x=3, holes_number_y=4),
        #             A.GridDropout(p=prob)], p=prob),
        #     A.HorizontalFlip(p=prob),
        #     A.GaussNoise(p=prob)
        # ])
        

        # Apply the transforms
        augmented = albumentations_transforms(image=img)
        img = augmented['image']
        
        #-- topil
        topil = torchvision.transforms.ToPILImage()
        img = topil(img)
        
        return img
        
        
    
    def __getitem__(self, index): 
        ID = self.list_IDs[index] 
        X1 = Image.open(self.train_path + ID + '/street.jpg').convert('RGB')
        
        X1 = self.augmentations(X1)
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
    
    



class Map_Dataset_v6(torch.utils.data.Dataset):
    
    '''
    
    v5 => v6 
    - topview photo pipe added  
    - topview resize function 
    - normalize pipe 

    '''      
    
    def __init__(self, list_IDs,train_path, max_size, cfg, split): 
        self.list_IDs = list_IDs
        self.train_path = train_path
        self.max_value = max_size
        self.min_value = int(self.max_value* 2/3) # min : max = 2 :3
        self.cfg = cfg
        self.split = split
    
    def __len__(self):
        return len(self.list_IDs)
    
    
    def resize(self,img_):
        if self.cfg.INTERPOLATION == "bilinear":
            inter_ = torchvision.transforms.InterpolationMode.BICUBIC
        elif self.cfg.INTERPOLATION == "bicubic":
            inter_ = torchvision.transforms.InterpolationMode.BILINEAR
        
        resize_transform = torchvision.transforms.Resize(
            size=self.min_value, max_size= self.max_value,
             interpolation=inter_ 
        )
        img_ = resize_transform(img_)
        return img_
    
    def resize_topview(self,img_):
        if self.cfg.INTERPOLATION == "bilinear":
            inter_ = torchvision.transforms.InterpolationMode.BICUBIC
        elif self.cfg.INTERPOLATION == "bicubic":
            inter_ = torchvision.transforms.InterpolationMode.BILINEAR
            
        resize_transform = torchvision.transforms.Resize(
            size = self.max_value,
            interpolation=inter_ 
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
    

    def normalize(self,img_):
        
        torchvision_transform = torchvision.transforms.Compose([

        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))           
        ])
        
        return torchvision_transform(img_)
    
        
    
    
    def ratio_pad(self,img_da):
        
        max_value = self.max_value       
        width_, height_ = img_da.size
        # left top right bottom 
        img_da = F.pad(img=img_da,padding=[(max_value-width_)//2+1, (max_value-height_)//2+1, (max_value-width_)//2+1,(max_value-height_)//2+1], padding_mode="constant", fill=0)    
        #img_da = F.pad(img=img_da,padding=[0, max_value-height_, max_value-width_,0], padding_mode="constant", fill=0)
        
        return img_da
    
    
    def augmentations(self,img):
        
        prob = 0.5
        AUGMENTATIONS = {
            "griddropout": lambda prob: A.OneOf([A.GridDropout(p=prob, holes_number_x=3, holes_number_y=4), A.GridDropout(p=prob)], p=prob),
            "horizontalflip": lambda prob: A.HorizontalFlip(p=prob),
            "gaussnoise": lambda prob: A.GaussNoise(p=prob)
        }
        
        img = np.asarray(img)

        # Create the list of transformations based on the configuration
        transforms_list = [AUGMENTATIONS[aug](prob) for aug in self.cfg.AUGMENTATIONS]

        # Compose the transformations
        albumentations_transforms = A.Compose(transforms_list)
        
        # Apply the transforms
        augmented = albumentations_transforms(image=img)
        img = augmented['image']
        
        #-- topil
        topil = torchvision.transforms.ToPILImage()
        img = topil(img)
        
        return img
        
        
    
    def __getitem__(self, index): 
        ID = self.list_IDs[index] 
        
        # X1 ---------------------------------------------
        X1 = Image.open(self.train_path + ID + '/street.jpg').convert('RGB')
        if self.split == 'train':
            X1 = self.augmentations(X1)
        elif self.split == 'valid':
            pass
        X1 = self.resize(X1)
        X1 = self.ratio_pad(X1)
        X1 = self.centercrop_normalize(X1)
        
        # X2 --------------------------------------------
        X2 = Image.open(self.train_path + ID + '/orthophoto.tif')
        X2 = self.resize_topview(X2)
        if self.split == 'train':
            X2 = self.augmentations(X2)
        elif self.split == 'valid':
            pass
        X2 = self.normalize(X2)

        # X3 --------------------------------------------
        #X3 = rasterio.open(self.train_path + ID + '/s2_l2a.tif').read() 
        #X3 = np.transpose(X3, [1, 2, 0]) 
        #X3 = self.transform(X3)
        
        y = int(open(self.train_path + ID + '/label.txt', "r").read())
        return X1, X2, y 
    


class Map_Dataset_v7(torch.utils.data.Dataset):
    
    '''
    v6 => v7
    - sentienl photo pipe added
    
    
    v5 => v6 
    - topview photo pipe added  
    - topview resize function 
    - normalize pipe 

    '''      
    
    def __init__(self, list_IDs,train_path, max_size, cfg, split): 
        self.list_IDs = list_IDs
        self.train_path = train_path
        self.max_value = max_size
        self.min_value = int(self.max_value* 2/3) # min : max = 2 :3
        self.cfg = cfg
        self.split = split
    
    def __len__(self):
        return len(self.list_IDs)
    
    
    def resize(self,img_):
        if self.cfg.INTERPOLATION == "bilinear":
            inter_ = torchvision.transforms.InterpolationMode.BICUBIC
        elif self.cfg.INTERPOLATION == "bicubic":
            inter_ = torchvision.transforms.InterpolationMode.BILINEAR
        
        resize_transform = torchvision.transforms.Resize(
            size=self.min_value, max_size= self.max_value,
             interpolation=inter_ 
        )
        img_ = resize_transform(img_)
        return img_
    
    def resize_topview(self,img_):
        if self.cfg.INTERPOLATION == "bilinear":
            inter_ = torchvision.transforms.InterpolationMode.BICUBIC
        elif self.cfg.INTERPOLATION == "bicubic":
            inter_ = torchvision.transforms.InterpolationMode.BILINEAR
            
        resize_transform = torchvision.transforms.Resize(
            size = self.max_value,
            interpolation=inter_ 
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
    

    def normalize(self,img_):
        
        torchvision_transform = torchvision.transforms.Compose([

        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))           
        ])
        
        return torchvision_transform(img_)
    
        
    
    
    def ratio_pad(self,img_da):
        
        max_value = self.max_value       
        width_, height_ = img_da.size
        # left top right bottom 
        img_da = F.pad(img=img_da,padding=[(max_value-width_)//2+1, (max_value-height_)//2+1, (max_value-width_)//2+1,(max_value-height_)//2+1], padding_mode="constant", fill=0)    
        #img_da = F.pad(img=img_da,padding=[0, max_value-height_, max_value-width_,0], padding_mode="constant", fill=0)
        
        return img_da
    
    
    def augmentations(self,img):
        
        prob = 0.5
        AUGMENTATIONS = {
            "griddropout": lambda prob: A.OneOf([A.GridDropout(p=prob, holes_number_x=3, holes_number_y=4), A.GridDropout(p=prob)], p=prob),
            "horizontalflip": lambda prob: A.HorizontalFlip(p=prob),
            "gaussnoise": lambda prob: A.GaussNoise(p=prob)
        }
        
        img = np.asarray(img)

        # Create the list of transformations based on the configuration
        transforms_list = [AUGMENTATIONS[aug](prob) for aug in self.cfg.AUGMENTATIONS]

        # Compose the transformations
        albumentations_transforms = A.Compose(transforms_list)
        
        # Apply the transforms
        augmented = albumentations_transforms(image=img)
        img = augmented['image']
        
        #-- topil
        topil = torchvision.transforms.ToPILImage()
        img = topil(img)
        
        return img
        
        
    
    def __getitem__(self, index): 
        ID = self.list_IDs[index] 
        
        # X1 ---------------------------------------------
        X1 = Image.open(self.train_path + ID + '/street.jpg').convert('RGB')
        if self.split == 'train':
            X1 = self.augmentations(X1)
        elif self.split == 'valid':
            pass
        X1 = self.resize(X1)
        X1 = self.ratio_pad(X1)
        X1 = self.centercrop_normalize(X1)
        
        # X2 --------------------------------------------
        X2 = Image.open(self.train_path + ID + '/orthophoto.tif')
        X2 = self.resize_topview(X2)
        if self.split == 'train':
            X2 = self.augmentations(X2)
        elif self.split == 'valid':
            pass
        X2 = self.normalize(X2)

        # X3 --------------------------------------------
        X3 = rasterio.open(self.train_path + ID + '/s2_l2a.tif').read() 
        # X3 = np.transpose(X3, [1, 2, 0])
        # X3 = X3[...,[3,2,1]]*3e-4
        # topil = torchvision.transforms.ToPILImage()
        # X3 = topil(X3)
        # X3 = self.resize_topview(X3)
        # X3 = self.normalize(X3)
        
        y = int(open(self.train_path + ID + '/label.txt', "r").read())
        
        return X1, X2, X3, y 
    
    