import numpy as np 
import matplotlib.pyplot as plt
#from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np 

import sys
import os 
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F 


def torch_display(img,batch):
    
    '''
    input shape : batch or not  
    
    torch.Size([batch, channel, 256, 256])
    
    '''
    if batch:
        img = img[0,:]
    else:
        pass
             
    img = img.permute(1,2,0)
    img = np.asarray(img)   
    
    fig_size= (10,10)
    plt.figure(figsize=fig_size)
    plt.imshow(img)
    


def label_display(label, n_class , nrows, ncols, channel_order, batch):
    '''
    if batch 
    eg. torch.Size([1, 3, 256, 256])
    one-hot encoding 된 경우 
        
    '''
    if batch:
        label = label[0,:]

    fig_size = (16,16)
    fig,axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
    assert(n_class == (nrows * ncols))
    #cnt = 0
    for cnt in range(n_class):
            
        if channel_order =="torch":
            label[cnt,:,:] 
            axs[cnt].imshow(label[cnt,:,:] )
            axs[cnt].set_title(f"class : {cnt}")
            #cnt += 1
        else:
            print("type def plz")
            
    plt.tight_layout()
    plt.show()
    


def apply_one_hot(mask, n_class,batch=True):
    if batch==True:
        mask = mask.to(torch.int64)
        mask = mask.squeeze(1)
        mask = torch.nn.functional.one_hot(mask, num_classes=n_class)
        mask = mask.permute(0,3,2,1)
        return mask
    else:
        mask = mask.to(torch.int64)
        #mask = mask.squeeze(0)
        mask = torch.nn.functional.one_hot(mask, num_classes=n_class)
        mask = mask.permute(0,3,2,1)
        return mask

    

def __mask_encoding__(self, label):
    
    label = np.asanyarray(label)
    
    zero_label = np.zeros((label.shape[0],label.shape[1],label.shape[2]))
    
    for k,v in self.palette.items():
        indices = np.where(np.all(label == v, axis=-1))
        zero_label[indices] = k 
    
    zero_label_ = zero_label[:,:,0].copy()

    label_oh = to_categorical(zero_label_,num_classes= len(self.palette.keys()) )     
    
    return label_oh



def label_to_edge(seg_map,edge_width):

    seg_map = np.asarray(seg_map)
    #print("label to edge shape : ", seg_map.shape)
    
    # shape :  (512, 512, 3)
    #seg_map = seg_map[:,:,1].copy()
    #print("shape : ", seg_map.shape)    
    
    h,w = seg_map.shape
    edge = np.zeros((h, w), dtype=np.uint8)
    
    # attach label to zero 
    #ignore_index = 255
    ignore_index = 999
    
    # down
    edge_down = edge[1:h, :]
    edge_down[(seg_map[1:h, :] != seg_map[:h - 1, :])
              & (seg_map[1:h, :] != ignore_index) &
              (seg_map[:h - 1, :] != ignore_index)] = 1
    # left
    edge_left = edge[:, :w - 1]
    edge_left[(seg_map[:, :w - 1] != seg_map[:, 1:w])
              & (seg_map[:, :w - 1] != ignore_index) &
              (seg_map[:, 1:w] != ignore_index)] = 1
    # up_left
    edge_upleft = edge[:h - 1, :w - 1]
    edge_upleft[(seg_map[:h - 1, :w - 1] != seg_map[1:h, 1:w])
                & (seg_map[:h - 1, :w - 1] != ignore_index) &
                (seg_map[1:h, 1:w] != ignore_index)] = 1
    # up_right
    edge_upright = edge[:h - 1, 1:w]
    edge_upright[(seg_map[:h - 1, 1:w] != seg_map[1:h, :w - 1])
                 & (seg_map[:h - 1, 1:w] != ignore_index) &
                 (seg_map[1:h, :w - 1] != ignore_index)] = 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (edge_width, edge_width))
    
    edge = cv2.dilate(edge, kernel)

    #--------
    #stack_edge = np.stack([edge, edge, edge], axis=-1)

    #print("dd ", stack_edge.shape)
    return edge




def robust_weight_average(model_org, model, alpha):
    #-------------------------------------------
    
    theta_0 = model_org.state_dict()
    theta_1 = model.state_dict()
    
    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())
    
    
    # interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }
    
    # update the model acccording to the new weights
    model.load_state_dict(theta)

    # evaluate
    #evaluate(finetuned, args)

    return model 



def save_current_file(EXEC_VER):
    current_file = sys.argv[0]
    file_path, file_name = os.path.split(current_file)
    
    target_path = os.path.join(file_path, "02.ckpts")
    new_file_path = os.path.join(target_path, f"ver_{EXEC_VER}_" + file_name)
    shutil.copyfile(current_file, new_file_path)

    print("#-----------------------------------------")
    print("File saved at:", new_file_path)
    
    

def resume_train(model, tgt_path):
#-- resume
    #tgt_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/05.Training/Segmentation/02.ckpts"
    print("#----------------------------------------")
    print("target path : ", tgt_path)
    ckpt_path = os.path.join( tgt_path, sorted(os.listdir(tgt_path))[-1]  )
    print("resume chekcpoint : ",ckpt_path)

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)
    
    return model
    


#-- metric 
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

        #dice score
        dice_coefficient = self._get_batch_dice_coefficient(onehot_pred, onehot_target)
        return loss1, loss2, dice_coefficient

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


#---

#-- metric 
class UNet_metric_v1():
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

        #dice score
        dice_coefficient = self._get_batch_dice_coefficient(onehot_pred, onehot_target)
        return loss1, loss2, dice_coefficient

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



def dice_coef(input, target, n_class, batch, bce_loss):
    

    #----------------------------------------
    # so important for calculating result
    # input = torch.argmax(input, dim=1)
    # input = F.one_hot(input, num_classes=n_class).permute(0, 3, 1, 2) 
    #----------------------------------------
    # background channel out
    if bce_loss != True:
        input =   input[:,1:,:,:]
        target = target[:,1:,:,:]
    else:
        pass
    #----------------------------------------
    
    
    smooth = 1.

    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    
    nominator = (2. * intersection + smooth)
    denominator =  (iflat.sum() + tflat.sum() + smooth)
    result = nominator / denominator
    
    return result
    
    
    

def iou_coef(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )



