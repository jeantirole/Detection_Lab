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
            
            mask_edge = RS_utils.label_to_edge(mask,3)
            mask_edge = torch.from_numpy(mask_edge).long()
            
            #-----
            mask = mask.unsqueeze(0).float()
            mask = torch.cat([mask, mask, mask], dim=0)
            #mask = mask.unsqueeze(0)
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



#------ 
class UNet_metric():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.CE_loss = nn.CrossEntropyLoss(reduction="mean") # "mean" or "sum"

    def __call__(self, pred, target):
        # cross-entropy
        loss1 = self.CE_loss(pred, target)
        
        # dice-coefficient
        onehot_pred = F.one_hot(torch.argmax(pred, dim=1), num_classes=self.num_classes).permute(0, 3, 1, 2) 
        onehot_target = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)
        loss2 = self._get_dice_loss(onehot_pred, onehot_target)
        
        # total loss
        total_loss = loss1 + loss2

        #dice score
        dice_coefficient = self._get_batch_dice_coefficient(onehot_pred, onehot_target)
        return total_loss, dice_coefficient

    def _get_dice_coeffient(self, pred, target):
        set_inter = torch.dot(pred.reshape(-1).float(), target.reshape(-1).float())
        set_sum = pred.sum() + target.sum()
        if set_sum.item() == 0:
            set_sum = 2 * set_inter
        dice_coeff = (2 * set_inter) / (set_sum + 1e-9)
        return dice_coeff

    def _get_multiclass_dice_coefficient(self, pred, target):
        dice = 0
        for class_index in range(1, self.num_classes):
            dice += self._get_dice_coeffient(pred[class_index], target[class_index])
        return dice / (self.num_classes - 1)

    def _get_batch_dice_coefficient(self, pred, target):
        num_batch = pred.shape[0]
        dice = 0
        for batch_index in range(num_batch):
            dice += self._get_multiclass_dice_coefficient(pred[batch_index], target[batch_index])
        return dice / num_batch

    def _get_dice_loss(self, pred, target):
        return 1 - self._get_batch_dice_coefficient(pred, target)