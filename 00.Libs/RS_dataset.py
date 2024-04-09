import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from albumentations import *
from torchvision import transforms
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