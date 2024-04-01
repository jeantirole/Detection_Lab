from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
from glob import glob
import os
from torchvision import transforms as T
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm 
import albumentations as A


import torchvision.transforms as transforms



ISAID_CLASSES = ('background', 'ship', 'store_tank', 'baseball_diamond',
               'tennis_court', 'basketball_court', 'Ground_Track_Field',
               'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
               'Harbor')

ISAID_PALETTE = {
    0: (0, 0, 0), 1: (0, 0, 63), 2: (0, 63, 63), 3: (0, 63, 0), 4: (0, 63, 127),
    5: (0, 63, 191), 6: (0, 63, 255), 7: (0, 127, 63), 8: (0, 127, 127),
    9: (0, 0, 127), 10: (0, 0, 191), 11: (0, 0, 255), 12: (0, 191, 127),
    13: (0, 127, 191), 14: (0, 127, 255), 15: (0, 100, 155)}





class SegDataset(torch.utils.data.Dataset):
    def __init__(self, imagesPath, masksPath, img_size,augmentations=None):
        self.imagesPath = imagesPath
        self.masksPath = masksPath
        self.img_size = img_size
        self.imagenet_default_mean = [0.485, 0.456, 0.406]
        self.imagenet_default_std = [0.229, 0.224, 0.225]

        self.augmentations = augmentations
        
        # img list
        self.img_list  =  sorted(glob(os.path.join(self.imagesPath,"*.png")) )
        # mask list 
        f_list = glob(os.path.join(self.masksPath ,"*.png") )
        D_ = {int(file_name.split('/')[-1].replace('.png', '')): file_name for file_name in f_list}
        self.mask_list = [ D_[k] for k in sorted(D_.keys()) ]
        
        print(self.img_list[0:4])
        print(self.mask_list[0:4])
        #self.mask_list = np.array( np.load(self.masksPath,allow_pickle=True),dtype=float)
        
        self.palette = ISAID_PALETTE
        

    
    def __label_totensor__(self, label):
        
        #label = np.asarray(label)
        label = to_categorical(label,num_classes=len(self.palette))
        #--- totensor
        label = torch.from_numpy(label).float()
        label = label.permute(2,0,1)
        
        #--- resize
        obj_label = T.Compose([
                T.Resize(size= (self.img_size,self.img_size),interpolation=InterpolationMode.NEAREST),
        ])
        
        label = obj_label(label)
        
        return label
    
    
    def __img_totensor__(self,x): 
               
        #--- totensor
        x = torch.from_numpy(x).float()
        x = x.permute(2,0,1)
        
        #--- resize
        obj_ = T.Compose([
        T.Resize(size=(self.img_size,self.img_size),interpolation=InterpolationMode.BILINEAR),
        T.Normalize(self.imagenet_default_mean, self.imagenet_default_std)
        ])
        r = obj_(x)
        
        return r
    
    
    def __len__(self):
        return len(self.imagesPath)

    def __aug__(self, image, mask):
        alb_transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ]
        augmented = A.Compose(alb_transform)
        tras = augmented(image=image, mask=mask)
        image = tras['image']
        mask  = tras['mask']
    
        return image, mask 
    
    
    def __getitem__(self, idx):
        image = self.img_list[idx]
        mask = self.mask_list[idx]
        
        # image ---
        image = Image.open(image)
        image = np.asarray(image)
               
        # mask ---
        mask = Image.open(mask)
        mask = np.asarray(mask)
                   
        # Apply transformations
        image, mask = self.__aug__(image, mask)
        
        # to tensor         
        image = self.__img_totensor__(image)
        mask = self.__label_totensor__(mask)
    
    
        return image, mask 
    
    
    