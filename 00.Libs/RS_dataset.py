import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from albumentations import *
import albumentations as A 
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from glob import glob
import segmentation_models_pytorch as smp

import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm 

import wandb
import logging
from tqdm import tqdm
import RS_utils


    
#----------------------------------------------------------------------------


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


ISAID_CLASSES_SHIP = (
    'background','ship','harbor' 
    )


ISAID_PALETTE_SHIP = {
    0: (0, 0, 0), 
    1: (0, 0, 63), 
    2: (0, 100, 155)}


#----------------------------------------------------------------------------

class Seg_RS_dataset():
    def __init__(self, img_dir,mask_dir, image_resize,phase,palette):
        self.phase = phase

        self.image_files = sorted(glob( os.path.join( img_dir , "*.png")) )
        self.mask_files  = sorted(glob( os.path.join( mask_dir , "*.png")))

        assert len(self.image_files) == len(self.mask_files)

        self.IMAGE_SIZE = image_resize #224 => 250
        
        self.palette = palette
        
        self.trans_ = Compose([ 
                RandomCrop(height=256,width=256, always_apply=True),
                # affine
                
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                #RandomBrightnessContrast(p=0.5),
                ])
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    

    def __len__(self):
        return len(self.image_files)

    def build_transformer_normalize(self):
      transformer = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      return transformer

    def convert_to_target_(self, mask_image, IMAGE_SIZE):
        #print(mask_image.shape)
        mask_image = np.asarray(mask_image)

        canvas = np.zeros( (mask_image.shape[0],mask_image.shape[1]) ,  dtype=np.uint8)
        for k,v in self.palette.items():
            canvas[np.all(mask_image == v, axis=-1)] = k

        #-------
        #mask = np.argmax(canvas, axis=-1 )
        #print(canvas.shape)

        return canvas

    def __getitem__(self, index):

        # image
        image = cv2.imread( self.image_files[index] )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            image = cv2.resize(image, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # mask
        mask = cv2.imread( self.mask_files[index] )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            mask = cv2.resize(mask, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) # inter nearest
        
        # build normalize
        normalizer = self.trans_normalizer

        if self.phase=='train':
            
            # aug 
            augmented = self.trans_(image=image,mask=mask)
            image= augmented['image']
            mask = augmented['mask']

            # processing 
            image = normalizer(image)
            mask = self.convert_to_target_(mask, self.IMAGE_SIZE)
            mask = torch.from_numpy(mask).long()
            
            return image, mask
    
        elif self.phase=='val':
          # normalize validation 할 때는 그냥 normalize 빼보자
           image = normalizer(image)

           mask = self.convert_to_target_2(mask_image, self.IMAGE_SIZE)
           target = torch.from_numpy(mask).long()
           return image, target
       
    def collate_fn(self,batch):
        images = []
        targets = []
        for a, b in batch:
            
            images.append(a)
            targets.append(b)
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)

        return images, targets

#------------------------------------------------------------------------------------------------------------------------

class Seg_RS_dataset_v1():
    def __init__(self, img_dir,mask_dir, image_resize,phase):
        self.phase = phase

        self.image_files = sorted(glob( os.path.join( img_dir , "*.png")) )
        self.mask_files  = sorted(glob( os.path.join( mask_dir , "*.png")))

        assert len(self.image_files) == len(self.mask_files)

        self.IMAGE_SIZE = image_resize #224 => 250

        self.ISAID_PALETTE = {
            0: (0, 0, 0), 1: (255, 255, 255) }

        self.ISAID_CLASSES = ('background', 'house')
        
        self.trans_ = Compose([
                RandomCrop(height=256,width=256, always_apply=True),
                # affine
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                #RandomRotate90(p=0.5),
                #RandomBrightnessContrast(p=0.5),
                ])
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    

    def __len__(self):
        return len(self.image_files)

    def build_transformer_normalize(self):
      transformer = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      return transformer

    def convert_to_target_(self, mask_image, IMAGE_SIZE):
        #print(mask_image.shape)
        mask_image = np.asarray(mask_image)

        canvas = np.zeros( (mask_image.shape[0],mask_image.shape[1]) ,  dtype=np.uint8)
        for k,v in self.ISAID_PALETTE.items():
            canvas[np.all(mask_image == v, axis=-1)] = k

        #-------
        #mask = np.argmax(canvas, axis=-1 )
        #print(canvas.shape)

        return canvas

    def __getitem__(self, index):

        # image
        image = cv2.imread( self.image_files[index] )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # mask
        mask = cv2.imread( self.mask_files[index] )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        #mask = cv2.resize(mask, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) # inter nearest
        
        # build normalize
        normalizer = self.trans_normalizer

        if self.phase=='train':
            
            # aug 
            augmented = self.trans_(image=image,mask=mask)
            image= augmented['image']
            mask = augmented['mask']

            # processing 
            image = normalizer(image)
            mask = self.convert_to_target_(mask, self.IMAGE_SIZE)
            mask = torch.from_numpy(mask).long()
            
            return image, mask
    
        elif self.phase=='val':
          # normalize validation 할 때는 그냥 normalize 빼보자
           image = normalizer(image)

           mask = self.convert_to_target_2(mask_image, self.IMAGE_SIZE)
           target = torch.from_numpy(mask).long()
           return image, target
       
    def collate_fn(self,batch):
        images = []
        targets = []
        for a, b in batch:
            
            images.append(a)
            targets.append(b)
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)

        return images, targets

#-----

class Seg_RS_dataset_ship():
    def __init__(self, img_dir,mask_dir, image_resize,phase,palette):
        self.phase = phase

        #self.image_files = sorted(glob( os.path.join( img_dir , "*.png")) )
        #self.mask_files  = sorted(glob( os.path.join( mask_dir , "*.png")))
        self.image_files = img_dir
        self.mask_files = mask_dir

        assert len(self.image_files) == len(self.mask_files)

        self.IMAGE_SIZE = image_resize #224 => 250
        
        self.palette = palette
        
        self.trans_ = Compose([ 
                RandomCrop(height=256,width=256, always_apply=True),
                # affine
                
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                #RandomBrightnessContrast(p=0.5),
                ])
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    

    def __len__(self):
        return len(self.image_files)

    def build_transformer_normalize(self):
      transformer = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      return transformer

    def convert_to_target_(self, mask_image, IMAGE_SIZE):
        #print(mask_image.shape)
        mask_image = np.asarray(mask_image)

        canvas = np.zeros( (mask_image.shape[0],mask_image.shape[1]) ,  dtype=np.uint8)
        for k,v in self.palette.items():
            canvas[np.all(mask_image == v, axis=-1)] = k

        #-------
        #mask = np.argmax(canvas, axis=-1 )
        #print(canvas.shape)

        return canvas

    def __getitem__(self, index):

        # image
        image = cv2.imread( self.image_files[index] )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            image = cv2.resize(image, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # mask
        mask = cv2.imread( self.mask_files[index] )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            mask = cv2.resize(mask, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) # inter nearest
        
        # build normalize
        normalizer = self.trans_normalizer

        if self.phase=='train':
            
            # aug 
            augmented = self.trans_(image=image,mask=mask)
            image= augmented['image']
            mask = augmented['mask']

            # processing 
            image = normalizer(image)
            mask = self.convert_to_target_(mask, self.IMAGE_SIZE)
            mask = torch.from_numpy(mask).long()
            
            return image, mask
    
        elif self.phase=='val':
          # normalize validation 할 때는 그냥 normalize 빼보자
           image = normalizer(image)

           mask = self.convert_to_target_2(mask_image, self.IMAGE_SIZE)
           target = torch.from_numpy(mask).long()
           return image, target
       
    def collate_fn(self,batch):
        images = []
        targets = []
        for a, b in batch:
            
            images.append(a)
            targets.append(b)
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)

        return images, targets


#-----

class Seg_RS_dataset_edge():
    def __init__(self, img_dir,mask_dir, image_resize,phase,palette,gaussian):
        self.phase = phase

        #self.image_files = sorted(glob( os.path.join( img_dir , "*.png")) )
        #self.mask_files  = sorted(glob( os.path.join( mask_dir , "*.png")))
        self.image_files = img_dir
        self.mask_files = mask_dir

        assert len(self.image_files) == len(self.mask_files)

        self.IMAGE_SIZE = image_resize #224 => 250
        
        self.palette = palette
        
        self.gaussian = gaussian
        
        self.trans_ = Compose([ 
                RandomCrop(height=256,width=256, always_apply=True),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                ])
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    

    def __len__(self):
        return len(self.image_files)

    def build_transformer_normalize(self):
      transformer = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      return transformer

    def convert_to_target_(self, mask_image, IMAGE_SIZE):
        #print(mask_image.shape)
        mask_image = np.asarray(mask_image)

        canvas = np.zeros( (mask_image.shape[0],mask_image.shape[1]) ,  dtype=np.uint8)
        for k,v in self.palette.items():
            canvas[np.all(mask_image == v, axis=-1)] = k

        #-------
        #mask = np.argmax(canvas, axis=-1 )
        #print(canvas.shape)

        return canvas

    def __getitem__(self, index):

        # image
        image = cv2.imread( self.image_files[index] )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            image = cv2.resize(image, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # mask
        mask = cv2.imread( self.mask_files[index] )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            mask = cv2.resize(mask, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) # inter nearest
        
        # build normalize
        normalizer = self.trans_normalizer

        if self.phase=='train':
            
            # aug 
            augmented = self.trans_(image=image,mask=mask)
            image= augmented['image']
            mask = augmented['mask']

            # processing 
            image = normalizer(image)
            
            mask = self.convert_to_target_(mask, self.IMAGE_SIZE)
            mask = torch.from_numpy(mask).long()
            
            mask_edge = RS_utils.label_to_edge(mask,3)
            mask_edge = torch.from_numpy(mask_edge).long()
            
            #-----
            mask = mask.unsqueeze(0).float()
            mask = torch.cat([mask, mask, mask], dim=0)
            #mask = mask.unsqueeze(0)
            #-----
            if self.gaussian:
                shape = mask.shape
                gaussian_noise = torch.normal(mean=0,std=0.1,size=shape)
                mask += gaussian_noise
            
            mask_edge = mask_edge.unsqueeze(0)
            
            return mask, mask_edge
    
        elif self.phase=='val':
          # normalize validation 할 때는 그냥 normalize 빼보자
           image = normalizer(image)
           mask = RS_utils.label_to_edge(mask,3)
           #mask = self.convert_to_target_2(mask_image, self.IMAGE_SIZE)
           target = torch.from_numpy(mask).long()
           return image, target
       
    def collate_fn(self,batch):
        images = []
        targets = []
        for a, b in batch:
            
            images.append(a)
            targets.append(b)
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)

        return images, targets
#----

class Seg_RS_dataset_edge_v1():
    def __init__(self, img_dir,mask_dir, image_resize,phase,palette,gaussian,mask_onehot,softmax):
        self.phase = phase
    
        #self.image_files = sorted(glob( os.path.join( img_dir , "*.png")) )
        #self.mask_files  = sorted(glob( os.path.join( mask_dir , "*.png")))
        self.image_files = img_dir
        self.mask_files = mask_dir

        assert len(self.image_files) == len(self.mask_files)

        self.IMAGE_SIZE = image_resize #224 => 250
        
        self.palette = palette
        
        self.gaussian = gaussian
        self.mask_onehot = mask_onehot
        self.softmax = softmax
        
        self.trans_ = Compose([ 
                RandomCrop(height=256,width=256, always_apply=True),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                ])
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    

    def __len__(self):
        return len(self.image_files)

    def build_transformer_normalize(self):
      transformer = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      return transformer

    def convert_to_target_(self, mask_image, IMAGE_SIZE):
        #print(mask_image.shape)
        mask_image = np.asarray(mask_image)

        canvas = np.zeros( (mask_image.shape[0],mask_image.shape[1]) ,  dtype=np.uint8)
        for k,v in self.palette.items():
            canvas[np.all(mask_image == v, axis=-1)] = k

        #-------
        #mask = np.argmax(canvas, axis=-1 )
        #print(canvas.shape)

        return canvas

    def __getitem__(self, index):

        # image
        image = cv2.imread( self.image_files[index] )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            image = cv2.resize(image, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # mask
        mask = cv2.imread( self.mask_files[index] )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            mask = cv2.resize(mask, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) # inter nearest
        
        # build normalize
        normalizer = self.trans_normalizer

        if self.phase=='train':
            
            # aug 
            augmented = self.trans_(image=image,mask=mask)
            image= augmented['image']
            mask = augmented['mask']

            # processing 
            image = normalizer(image)
            
            mask = self.convert_to_target_(mask, self.IMAGE_SIZE)
            mask = torch.from_numpy(mask).long()
            
            mask_edge = RS_utils.label_to_edge(mask,3)
            mask_edge = torch.from_numpy(mask_edge).long()
            
            #-----
            mask = mask.unsqueeze(0).float()
            #-----
            if self.mask_onehot:
                mask = mask.to(torch.int64)
                mask = mask.squeeze(0)
                mask = torch.nn.functional.one_hot(mask, num_classes=len(self.palette.keys()))
                mask = mask.permute(2,0,1)              
                
            
            if self.gaussian:
                mask = mask.float()
                shape = mask.shape
                gaussian_noise = torch.normal(mean=0,std=0.1,size=shape)
                mask += gaussian_noise
                
            if self.softmax:
                mask = torch.softmax(mask, dim=0) 
            
            mask_edge = mask_edge.unsqueeze(0)
            
            return mask, mask_edge
    
        elif self.phase=='val':
          # normalize validation 할 때는 그냥 normalize 빼보자
           image = normalizer(image)
           mask = RS_utils.label_to_edge(mask,3)
           #mask = self.convert_to_target_2(mask_image, self.IMAGE_SIZE)
           target = torch.from_numpy(mask).long()
           return image, target
       
    def collate_fn(self,batch):
        images = []
        targets = []
        for a, b in batch:
            
            images.append(a)
            targets.append(b)
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)

        return images, targets



class Seg_RS_dataset_edge_v2():
    def __init__(self, img_dir,mask_dir, image_resize,phase,palette,gaussian,mask_onehot,softmax):
        self.phase = phase
    
        #self.image_files = sorted(glob( os.path.join( img_dir , "*.png")) )
        #self.mask_files  = sorted(glob( os.path.join( mask_dir , "*.png")))
        self.image_files = img_dir
        self.mask_files = mask_dir

        assert len(self.image_files) == len(self.mask_files)

        self.IMAGE_SIZE = image_resize #224 => 250
        
        self.palette = palette
        
        self.gaussian = gaussian
        self.mask_onehot = mask_onehot
        self.softmax = softmax
        
        self.trans_ = Compose([ 
                RandomCrop(height=256,width=256, always_apply=True),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                ])
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    

    def __len__(self):
        return len(self.image_files)

    def build_transformer_normalize(self):
      transformer = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      return transformer

    def convert_to_target_(self, mask_image, IMAGE_SIZE):
        #print(mask_image.shape)
        mask_image = np.asarray(mask_image)

        canvas = np.zeros( (mask_image.shape[0],mask_image.shape[1]) ,  dtype=np.uint8)
        for k,v in self.palette.items():
            canvas[np.all(mask_image == v, axis=-1)] = k

        #-------
        #mask = np.argmax(canvas, axis=-1 )
        #print(canvas.shape)

        return canvas
    
    def _apply_one_hot_v1(self, mask):
        mask = mask.to(torch.int64)
        #mask = mask.squeeze(0)
        mask = torch.nn.functional.one_hot(mask, num_classes=len(self.palette.keys()))
        mask = mask.permute(0,3,2,1)
        return mask

    def _apply_one_hot(self, mask):
        mask = mask.to(torch.int64)
        mask = mask.squeeze(0)
        mask = torch.nn.functional.one_hot(mask, num_classes=len(self.palette.keys()))
        mask = mask.permute(2, 0, 1)
        return mask

    def _apply_gaussian(self, mask):
        mask = mask.float()
        shape = mask.shape
        gaussian_noise = torch.normal(mean=0, std=0.1, size=shape)
        mask += gaussian_noise
        return mask

    def _apply_softmax(self, mask):
        mask = mask.float()
        mask = torch.softmax(mask, dim=0)
        #mask = torch.sigmoid(mask)
        return mask

    def __getitem__(self, index):

        # image
        image = cv2.imread( self.image_files[index] )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            image = cv2.resize(image, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # mask
        mask = cv2.imread( self.mask_files[index] )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            mask = cv2.resize(mask, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) # inter nearest
        
        # build normalize
        normalizer = self.trans_normalizer

        if self.phase=='train':
            
            # aug 
            augmented = self.trans_(image=image,mask=mask)
            image= augmented['image']
            mask = augmented['mask']

            # processing 
            image = normalizer(image)
            
            mask = self.convert_to_target_(mask, self.IMAGE_SIZE)
            mask = torch.from_numpy(mask).long()
            original_mask = mask.clone()


            mask_edge = RS_utils.label_to_edge(mask,3)
            mask_edge = torch.from_numpy(mask_edge).long()
            
            
            #mask_edge = mask_edge.unsqueeze(0)


            #-----
            mask = mask.unsqueeze(0).float()
            #-----
            if self.mask_onehot:
                mask = self._apply_one_hot(mask)

            if self.gaussian:
                mask = self._apply_gaussian(mask)

            if self.softmax:
                mask = self._apply_softmax(mask)
        
            
            return image, original_mask, mask
    
        elif self.phase=='val':
          # normalize validation 할 때는 그냥 normalize 빼보자
           image = normalizer(image)
           mask = RS_utils.label_to_edge(mask,3)
           #mask = self.convert_to_target_2(mask_image, self.IMAGE_SIZE)
           target = torch.from_numpy(mask).long()
           return image, target
       
    def collate_fn(self,batch):
        images = []
        targets = []
        targets_edges = []
        for a, b, c in batch:
            
            images.append(a)
            targets.append(b)
            targets_edges.append(c)
        images = torch.stack(images, dim=0)
        original_mask = torch.stack(targets, dim=0)
        mask = torch.stack(targets_edges, dim=0)

        return images, original_mask, mask

    
#------ v3

class Seg_RS_dataset_edge_v3():
    def __init__(self, img_dir,mask_dir, image_resize,phase,palette,gaussian,mask_onehot,softmax):
        self.phase = phase
    
        #self.image_files = sorted(glob( os.path.join( img_dir , "*.png")) )
        #self.mask_files  = sorted(glob( os.path.join( mask_dir , "*.png")))
        self.image_files = img_dir
        self.mask_files = mask_dir

        assert len(self.image_files) == len(self.mask_files)

        self.IMAGE_SIZE = image_resize #224 => 250
        
        self.palette = palette
        
        self.gaussian = gaussian
        self.mask_onehot = mask_onehot
        self.softmax = softmax
        
        self.trans_ = Compose([ 
                RandomCrop(height=256,width=256, always_apply=True),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                ])
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    

    def __len__(self):
        return len(self.image_files)

    def build_transformer_normalize(self):
      transformer = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      return transformer

    def convert_to_target_(self, mask_image, IMAGE_SIZE):
        #print(mask_image.shape)
        mask_image = np.asarray(mask_image)

        canvas = np.zeros( (mask_image.shape[0],mask_image.shape[1]) ,  dtype=np.uint8)
        for k,v in self.palette.items():
            canvas[np.all(mask_image == v, axis=-1)] = k

        #-------
        #mask = np.argmax(canvas, axis=-1 )
        #print(canvas.shape)

        return canvas
    
    def _apply_one_hot_v1(self, mask):
        mask = mask.to(torch.int64)
        #mask = mask.squeeze(0)
        mask = torch.nn.functional.one_hot(mask, num_classes=len(self.palette.keys()))
        mask = mask.permute(2,1,0)
        return mask
    
    def _apply_one_hot(self, mask):
        mask = mask.to(torch.int64)
        mask = mask.squeeze(0)
        mask = torch.nn.functional.one_hot(mask, num_classes=len(self.palette.keys()))
        mask = mask.permute(2, 0, 1)
        return mask

    def _apply_gaussian(self, mask):
        mask = mask.float()
        shape = mask.shape
        gaussian_noise = torch.normal(mean=0, std=0.1, size=shape)
        mask += gaussian_noise
        return mask

    def _apply_softmax(self, mask):
        mask = mask.float()
        mask = torch.softmax(mask, dim=0)
        #mask = torch.sigmoid(mask)
        return mask

    def __getitem__(self, index):

        # image
        image = cv2.imread( self.image_files[index] )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            image = cv2.resize(image, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # mask
        mask = cv2.imread( self.mask_files[index] )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            mask = cv2.resize(mask, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) # inter nearest
        
        # build normalize
        normalizer = self.trans_normalizer

        if self.phase=='train':
            
            # aug 
            augmented = self.trans_(image=image,mask=mask)
            image= augmented['image']
            mask = augmented['mask']

            # processing 
            image = normalizer(image)
            
            mask = self.convert_to_target_(mask, self.IMAGE_SIZE)
            mask = torch.from_numpy(mask).long()
            original_mask = mask.clone()
            #original_mask = original_mask.unsqueeze(1)
            original_mask = self._apply_one_hot_v1(original_mask)


            
            mask_edge = RS_utils.label_to_edge(mask,3)
            mask_edge = torch.from_numpy(mask_edge).long()
            mask_edge = mask_edge.unsqueeze(0)


            #-----
            mask = mask.unsqueeze(0).float()
            #-----
            if self.mask_onehot:
                mask = self._apply_one_hot(mask)

            if self.gaussian:
                mask = self._apply_gaussian(mask)

            if self.softmax:
                mask = self._apply_softmax(mask)
        
            
            return image, original_mask, mask, mask_edge
    
        elif self.phase=='val':
          # normalize validation 할 때는 그냥 normalize 빼보자
           image = normalizer(image)
           mask = RS_utils.label_to_edge(mask,3)
           #mask = self.convert_to_target_2(mask_image, self.IMAGE_SIZE)
           target = torch.from_numpy(mask).long()
           return image, target
       
    def collate_fn(self,batch):
        images = []
        targets = []
        targets_edges = []
        mask_edges = []
        for a, b, c, d in batch:
            
            images.append(a)
            targets.append(b)
            targets_edges.append(c)
            mask_edges.append(d)
        images = torch.stack(images, dim=0)
        original_mask = torch.stack(targets, dim=0)
        mask = torch.stack(targets_edges, dim=0)
        mask_edges = torch.stack(mask_edges, dim=0)
    
        return images, original_mask, mask, mask_edges
    
    
    
    


class Seg_RS_dataset_edge_v4():
    def __init__(self, img_dir,mask_dir, image_resize,phase,palette):
        self.phase = phase
        self.image_files = img_dir
        self.mask_files = mask_dir

        assert len(self.image_files) == len(self.mask_files)

        self.IMAGE_SIZE = image_resize #224 => 250
        
        self.palette = palette
        
        self.trans_ = Compose([ 
                RandomCrop(height=256,width=256, always_apply=True),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                ])
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    

    def __len__(self):
        return len(self.image_files)

    def build_transformer_normalize(self):
        transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        return transformer

    def convert_to_target_(self, mask_image, IMAGE_SIZE):
        #print(mask_image.shape)
        mask_image = np.asarray(mask_image)

        canvas = np.zeros( (mask_image.shape[0],mask_image.shape[1]) ,  dtype=np.uint8)
        for k,v in self.palette.items():
            canvas[np.all(mask_image == v, axis=-1)] = k

        return canvas
    
    def mask_onehot(self, mask):
        mask = mask.to(torch.int64)
        mask = mask.squeeze(0)
        mask = torch.nn.functional.one_hot(mask, num_classes=len(self.palette.keys()))
        mask = mask.permute(2,0,1)
        
        return mask

    def __getitem__(self, index):

        # image
        image = cv2.imread( self.image_files[index] )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            image = cv2.resize(image, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # mask
        mask = cv2.imread( self.mask_files[index] )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            mask = cv2.resize(mask, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) # inter nearest
        
        # build normalize
        normalizer = self.trans_normalizer

        if self.phase=='train':
            
            #----- aug 
            augmented = self.trans_(image=image,mask=mask)
            image= augmented['image']
            mask = augmented['mask']

            #----- processing 
            image = normalizer(image)
            
            mask = self.convert_to_target_(mask, self.IMAGE_SIZE)
            mask = torch.from_numpy(mask).long()     
            
            mask_edge = RS_utils.label_to_edge(mask,3)
            mask_edge = torch.from_numpy(mask_edge) # edge type is important for BCE, target should be long. ? or float ? 
            
            #----- onehot decision
            if self.mask_onehot:
                mask = self.mask_onehot(mask)     
            
            
            #----- type decision 
            mask = mask.float()              
            mask_edge = mask_edge.unsqueeze(0).float()
            
            return mask, mask_edge
    
        elif self.phase=='val':
            # normalize validation 할 때는 그냥 normalize 빼보자
            image = normalizer(image)
            mask = RS_utils.label_to_edge(mask,3)
            #mask = self.convert_to_target_2(mask_image, self.IMAGE_SIZE)
            target = torch.from_numpy(mask).long()
            return image, target
    
    def collate_fn(self,batch):
        images = []
        targets = []
        for a, b in batch:
            
            images.append(a)
            targets.append(b)
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)

        return images, targets
    
    

class Seg_RS_dataset_edge_v5():
    def __init__(self, 
                 img_dir,
                 mask_dir, 
                 image_resize,
                 crop_size,
                 phase,
                 palette,
                 edge_return,
                 mask_onehot):
        
        self.phase = phase
        self.image_files = img_dir
        self.mask_files = mask_dir
        self.edge_return = edge_return
        self.mask_onehot_return = mask_onehot

        assert len(self.image_files) == len(self.mask_files)

        self.IMAGE_SIZE = image_resize #224 => 250
        
        self.palette = palette
        
        self.trans_ = Compose([ 
                RandomCrop(height=crop_size,width=crop_size, always_apply=True),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                ])
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    

    def __len__(self):
        return len(self.image_files)

    def build_transformer_normalize(self):
        transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        return transformer

    def convert_to_target_(self, mask_image, IMAGE_SIZE):
        #print(mask_image.shape)
        mask_image = np.asarray(mask_image)

        canvas = np.zeros( (mask_image.shape[0],mask_image.shape[1]) ,  dtype=np.uint8)
        for k,v in self.palette.items():
            canvas[np.all(mask_image == v, axis=-1)] = k

        return canvas
    
    def mask_onehot(self, mask):
        mask = mask.to(torch.int64)
        mask = mask.squeeze(0)
        mask = torch.nn.functional.one_hot(mask, num_classes=len(self.palette.keys()))
        mask = mask.permute(2,0,1)
        
        return mask

    def __getitem__(self, index):

        # image
        image = cv2.imread( self.image_files[index] )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            image = cv2.resize(image, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # mask
        mask = cv2.imread( self.mask_files[index] )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.IMAGE_SIZE != None:
            mask = cv2.resize(mask, dsize=(self.IMAGE_SIZE,self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) # inter nearest
        
        # build normalize
        normalizer = self.trans_normalizer

        if self.phase=='train':
                
            #----- aug 
            augmented = self.trans_(image=image,mask=mask)
            image= augmented['image']
            mask = augmented['mask']

            #----- processing 
            image = normalizer(image)
            
            mask = self.convert_to_target_(mask, self.IMAGE_SIZE)
            mask = torch.from_numpy(mask).long()     
            
            if self.edge_return:
                mask_edge = RS_utils.label_to_edge(mask,3)
                mask_edge = torch.from_numpy(mask_edge) # edge type is important for BCE, target should be long. ? or float ? 
                mask_edge = mask_edge.unsqueeze(0).float()    
            
            #----- onehot decision
            if self.mask_onehot_return:
                mask = self.mask_onehot(mask)     
            
            
            #----- type decision 
            # mask = mask.float()              
            # cross entropy needs scalar 
            
            return image, mask
            
    
    def collate_fn(self,batch):
        images = []
        targets = []
        for a, b in batch:
            
            images.append(a)
            targets.append(b)
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)

        return images, targets
    
    
    



# Object Detection Dataset 

from torchvision.datasets import VOCDetection
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import collections

class myVOCDetection(VOCDetection):
    
    def build_transformer_normalize(self):
        transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        return transformer
    
    
    
    def __getitem__(self, index):
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
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot()) 

        targets = [] # bbox coordinates 
        labels = [] # bbox class 

        # get bbox info 
        for t in target['annotation']['object']:
            label = np.zeros(5)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox']['ymax'], classes.index(t['name'])

            targets.append(list(label[:4])) # bbox coord
            labels.append(label[4])         # bboc class

        if self.transforms:
            augmentations = self.transforms(image=img, bboxes=targets)
            img = augmentations['image']
            targets = augmentations['bboxes']

        #-- normalizer 
        normalizer = self.build_transformer_normalize()
        img = normalizer(img)
        
        #-- padding for bboxes 
        
        

        return img, targets, labels
    
    
    def collate_fn(self,batch):
        images = []
        targets = []
        classes_ = []
        for a, b, c in batch:
            
            images.append(a)
            targets.append(b)
            classes_.append(c)
            
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)
        classes_ = torch.stack(classes_, dim=0)

        return images, targets, classes_

    def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]: # xml => dict
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    





class Seg_OD_dataset_v1():
    def __init__(self, 
                 images,
                 annotations,):
        
        self.images = images
        self.annotations = annotations
        self.crop_size = None
        self.image_size = None

        assert len(self.images) == len(self.annotations)
        
        self.trans_ = Compose([ 
                RandomCrop(height=self.crop_size,width=self.crop_size, always_apply=True),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                ])
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    

    def __len__(self):
        return len(self.image_files)



    def __getitem__(self, index):

        # image
        image = cv2.imread( self.images[index] )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_size != None:
            image = cv2.resize(image, dsize=(self.image_size,self.image_size), interpolation=cv2.INTER_LINEAR)

        # mask
        mask = cv2.imread( self.annotations[index] )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.image_size != None:
            mask = cv2.resize(mask, dsize=(self.image_size,self.image_size), interpolation=cv2.INTER_NEAREST) # inter nearest
        
        # build normalize
        normalizer = self.trans_normalizer

        if self.phase=='train':
                
            #----- aug 
            augmented = self.trans_(image=image,mask=mask)
            image= augmented['image']
            mask = augmented['mask']

            #----- processing 
            image = normalizer(image)
            
            mask = self.convert_to_target_(mask, self.IMAGE_SIZE)
            mask = torch.from_numpy(mask).long()     
            
            if self.edge_return:
                mask_edge = RS_utils.label_to_edge(mask,3)
                mask_edge = torch.from_numpy(mask_edge) # edge type is important for BCE, target should be long. ? or float ? 
                mask_edge = mask_edge.unsqueeze(0).float()    
            
            #----- onehot decision
            if self.mask_onehot_return:
                mask = self.mask_onehot(mask)     
            
            
            #----- type decision 
            # mask = mask.float()              
            # cross entropy needs scalar 
            
            return image, mask
            
    
    def collate_fn(self,batch):
        images = []
        targets = []
        for a, b in batch:
            
            images.append(a)
            targets.append(b)
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)

        return images, targets
    

class ObjectDetectionDataset(Dataset):
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.
    
    ref : https://towardsdatascience.com/understanding-and-implementing-faster-r-cnn-a-step-by-step-guide-11acfff216b0
    
    Returns
    ------------
    images: torch.Tensor of size (B, C, H, W)
    gt bboxes: torch.Tensor of size (B, max_objects, 4)
    gt classes: torch.Tensor of size (B, max_objects)
    '''
    def __init__(self, annotation_path, img_dir, img_size, name2idx):
        self.annotation_path = annotation_path
        self.img_dir = img_dir
        self.img_size = img_size
        self.name2idx = name2idx
        
        self.img_data_all, self.gt_bboxes_all, self.gt_classes_all = self.get_data()
        
    def __len__(self):
        return self.img_data_all.size(dim=0)
    
    def __getitem__(self, idx):
        return self.img_data_all[idx], self.gt_bboxes_all[idx], self.gt_classes_all[idx]
        
    def get_data(self):
        img_data_all = []
        gt_idxs_all = []
        
        gt_boxes_all, gt_classes_all, img_paths = parse_annotation(self.annotation_path, self.img_dir, self.img_size)
        
        for i, img_path in enumerate(img_paths):
            
            # skip if the image path is not valid
            if (not img_path) or (not os.path.exists(img_path)):
                continue
                
            # read and resize image
            img = io.imread(img_path)
            img = resize(img, self.img_size)
            
            # convert image to torch tensor and reshape it so channels come first
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            
            # encode class names as integers
            gt_classes = gt_classes_all[i]
            gt_idx = torch.Tensor([self.name2idx[name] for name in gt_classes])
            
            img_data_all.append(img_tensor)
            gt_idxs_all.append(gt_idx)
        
        # pad bounding boxes and classes so they are of the same size
        gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=-1)
        
        # stack all images
        img_data_stacked = torch.stack(img_data_all, dim=0)
        
        return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad
    
    


class Detection_Dataset_v1(Dataset):
    
    def __init__(self, images, annotations, classes, image_size):
        self.images = sorted(glob(os.path.join(images,"*")))
        self.annotations = sorted(glob(os.path.join(annotations,"*")))
        self.classes = classes
        
        self.trans_normalizer  = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])])
        

        self.image_size = image_size
        self.scale = 1.0

        # 이미지에 padding을 적용하여 종횡비를 유지시키면서 크기가 600x600 되도록 resize 합니다.
        self.train_transforms = A.Compose([
                            A.LongestMaxSize(max_size=int(self.image_size * self.scale)),
                            A.PadIfNeeded(min_height=int(self.image_size*self.scale), min_width=int(self.image_size*self.scale),
                                        border_mode=cv2.BORDER_CONSTANT),
                            A.pytorch.ToTensorV2()],
                            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                            )

        self.val_transforms = A.Compose([
                            A.LongestMaxSize(max_size=int(self.image_size * self.scale)),
                            A.PadIfNeeded(min_height=int(self.image_size*self.scale), min_width=int(self.image_size*self.scale),border_mode=cv2.BORDER_CONSTANT),
                            A.pytorch.ToTensorV2()],
                            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                            )
        
        self.transform = True
        
        
    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]: # xml => dict
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    def __getitem__(self, index):

        img = np.array(Image.open(self.images[index]).convert('RGB'))
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot()) 

        targets = [] # bbox coordinates 
        labels = [] # bbox class 

        # get bbox info 
        for t in target['annotation']['object']:
            label = np.zeros(5)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox']['ymax'], self.classes.index(t['name'])

            targets.append(list(label[:4])) # bbox coord
            labels.append(label[4])         # bboc class


        if self.transform:
            augmentations = self.train_transforms(image=img, bboxes=targets)
            img = augmentations['image']
            targets = augmentations['bboxes']

        #-- normalizer
        img = img.float()
        img = self.trans_normalizer(img)
        
        #-- box, label to tensor 
        targets = torch.tensor(targets)
        labels = torch.tensor(labels)
        
        return img, targets, labels
    
    def collate_fn(self,batch):
        img_data_all = []
        gt_boxes_all = []
        gt_idxs_all = []
        
        #gt_boxes_all, gt_classes_all, img_paths = parse_annotation(self.annotation_path, self.img_dir, self.img_size)
        for a,b,c in batch:
            #---
            img_data_all.append(a)
            gt_boxes_all.append(b)
            gt_idxs_all.append(c)
        
        # pad bounding boxes and classes so they are of the same size
        gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=-1)
        
        # stack all images
        img_data_stacked = torch.stack(img_data_all, dim=0)
        
        return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad
    



#----------------------


class Detection_Dataset_v2(Dataset):
    
    def __init__(self, images, annotations, classes, image_size, normalizing):
        self.images = sorted(glob(os.path.join(images,"*")))
        self.annotations = sorted(glob(os.path.join(annotations,"*")))
        self.classes = classes
        self.normalizing = normalizing
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(), # tensor 로 변환하면서 0~1 범위로 noramlize 된다.
                # 최소값(=-1)은 (0 - 0.5) / 0.5 = -1, 최대값(=1) 은 (1 - 0.5) / 0.5 = 1 로 조정.
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
                ])
        

        self.image_size = image_size
        self.scale = 1.0

        # 이미지에 padding을 적용하여 종횡비를 유지시키면서 크기가 600x600 되도록 resize 합니다.
        # padding 종횡비가 문제였을까? 모델이 제대로 학습되지 않았음. 
        self.train_transforms = A.Compose([
                            A.Resize(self.image_size,self.image_size,interpolation=cv2.INTER_LINEAR),
                            A.HorizontalFlip(p=0.5)],
                            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                            )

        self.val_transforms = A.Compose([
                            A.Resize(self.image_size,self.image_size,interpolation=cv2.INTER_LINEAR),
                            A.HorizontalFlip(p=0.5)],
                            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                            )
        
        self.transform = True
        
        
    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]: # xml => dict
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    def __getitem__(self, index):
        
        img = cv2.imread( self.images[index] )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot()) 

        targets = [] # bbox coordinates 
        labels = [] # bbox class 

        # get bbox info 
        for t in target['annotation']['object']:
            label = np.zeros(5)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox']['ymax'], self.classes.index(t['name'])

            targets.append(list(label[:4])) # bbox coord
            labels.append(label[4])         # bboc class


        if self.transform:
            augmentations = self.train_transforms(image=img, bboxes=targets)
            img = augmentations['image']
            targets = augmentations['bboxes']

        #-- build normalize
        if self.normalizing:
            normalizer = self.trans_normalizer
            img = normalizer(img)
            
        
        #-- box, label to tensor 
        targets = torch.tensor(targets)
        labels = torch.tensor(labels)
        
        return img, targets, labels
    
    def collate_fn(self,batch):
        img_data_all = []
        gt_boxes_all = []
        gt_idxs_all = []
        
        #gt_boxes_all, gt_classes_all, img_paths = parse_annotation(self.annotation_path, self.img_dir, self.img_size)
        for a,b,c in batch:
            #---
            img_data_all.append(a)
            gt_boxes_all.append(b)
            gt_idxs_all.append(c)
        
        # pad bounding boxes and classes so they are of the same size
        gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=-1)
        
        # stack all images
        img_data_stacked = torch.stack(img_data_all, dim=0)
        
        return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad
    
    
    
#------
# Face Detection Dataset 

class MaskDataset(object):
    def __init__(self, transforms, path):
        '''
        path: path to train folder or test folder
        '''
        # transform module과 img path 경로를 정의
        self.transforms = transforms
        self.path = path
        self.imgs = list(sorted(os.listdir(self.path)))


    def __getitem__(self, idx): #special method
        # load images ad masks
        file_image = self.imgs[idx]
        file_label = self.imgs[idx][:-3] + 'xml'
        img_path = os.path.join(self.path, file_image)
        
        if 'test' in self.path:
            label_path = os.path.join("/mnt/hdd/eric/.tmp_ipy/00.Data/FaceMaskDetection/test_annotations/", file_label)
        else:
            label_path = os.path.join("/mnt/hdd/eric/.tmp_ipy/00.Data/FaceMaskDetection/annotations/", file_label)

        img = Image.open(img_path).convert("RGB")
        #Generate Label
        target = RS_utils.generate_target(label_path)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self): 
        return len(self.imgs)
    



#-------


class Detection_Dataset_v3(Dataset):
    
    def __init__(self, images, annotations, classes, image_size, normalizing):
        self.images = sorted(glob(os.path.join(images,"*")))
        self.annotations = sorted(glob(os.path.join(annotations,"*")))
        self.classes = classes
        self.normalizing = normalizing
        
        self.trans_normalizer  = transforms.Compose([
                transforms.ToTensor(), # tensor 로 변환하면서 0~1 범위로 noramlize 된다.
                # 최소값(=-1)은 (0 - 0.5) / 0.5 = -1, 최대값(=1) 은 (1 - 0.5) / 0.5 = 1 로 조정.
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
                ])
        

        self.image_size = image_size
        self.scale = 1.0

        # 이미지에 padding을 적용하여 종횡비를 유지시키면서 크기가 600x600 되도록 resize 합니다.
        # padding 종횡비가 문제였을까? 모델이 제대로 학습되지 않았음. 
        self.train_transforms = A.Compose([
                            A.Resize(self.image_size,self.image_size,interpolation=cv2.INTER_LINEAR),
                            A.HorizontalFlip(p=0.5)],
                            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                            )

        self.val_transforms = A.Compose([
                            A.Resize(self.image_size,self.image_size,interpolation=cv2.INTER_LINEAR),
                            A.HorizontalFlip(p=0.5)],
                            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                            )
        
        self.transform = True
        
        
    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]: # xml => dict
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    def __getitem__(self, index):
        
        img = cv2.imread( self.images[index] )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot()) 

        targets = [] # bbox coordinates 
        labels = [] # bbox class 

        # get bbox info 
        for t in target['annotation']['object']:
            label = np.zeros(5)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox']['ymax'], self.classes.index(t['name'])

            targets.append(list(label[:4])) # bbox coord
            labels.append(label[4])         # bboc class


        if self.transform:
            augmentations = self.train_transforms(image=img, bboxes=targets)
            img = augmentations['image']
            targets = augmentations['bboxes']

        #-- build normalize
        if self.normalizing:
            normalizer = self.trans_normalizer
            img = normalizer(img)
            
        
        #-- box, label to tensor         
        targets = torch.as_tensor(targets, dtype=torch.float32) 
        labels = torch.as_tensor(labels, dtype=torch.int64) 
        
        target_ = {}
        target_["boxes"] = targets
        target_["labels"] = labels
        
        
        return img, target_
    
    def collate_fn(self,batch):        
        # def collate_fn(self,batch):
        # img_data_all = []
        # gt_boxes_all = []
        # gt_idxs_all = []
        
        # #gt_boxes_all, gt_classes_all, img_paths = parse_annotation(self.annotation_path, self.img_dir, self.img_size)
        # for a,b,c in batch:
        #     #---
        #     img_data_all.append(a)
        #     gt_boxes_all.append(b)
        #     gt_idxs_all.append(c)
        
        # # pad bounding boxes and classes so they are of the same size
        # gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
        # gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=-1)
        
        # # stack all images
        # img_data_stacked = torch.stack(img_data_all, dim=0)
        
        # return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad
        return tuple(zip(*batch))
    