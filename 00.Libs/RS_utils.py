import numpy as np 
import matplotlib.pyplot as plt
#from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np 
import math

import sys
import os 
import shutil
import datetime
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F 


from glob import glob
from torch.nn.utils.rnn import pad_sequence
from albumentations.pytorch.transforms import ToTensorV2

def torch_denormalize(img):
    
    img[0,:]  =  (img[0,:]* 0.229) + 0.485
    img[1,:]  =  (img[1,:]* 0.224) + 0.456
    img[2,:]  =  (img[2,:]* 0.225) + 0.406
    
    return img   




def torch_display(image,batch,denormal):
    
    '''
    input shape : batch or not  
    
    torch.Size([batch, channel, 256, 256])
    
    '''
    img = image.clone() 
    if batch:
        img = img[0,:]
    else:
        pass
    
    #----- denormalizing func
    if denormal:
        # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img = torch_denormalize(img)
    #-----
             
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
    RS_utils.label_display(qa, n_class = sample_mask.shape[0], nrows= int(np.ceil(sample_mask.shape[0] / 2)) , ncols = 2 , channel_order='torch', batch=False)
        
    '''
    if batch:
        label = label[0,:]

    fig_size = (16,16)
    fig,axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
    #assert(n_class == (nrows * ncols))
    #cnt = 0
    axs = axs.flatten()

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



def dice_coef(input, target, n_class, batch, loss_type):
    

    #----------------------------------------
    # so important for calculating result
    # more options to be built
    # - each channel calculation 
    #----------------------------------------
    
    if loss_type == "CE":
        input = torch.argmax(input, dim=1)
        input = F.one_hot(input, num_classes=n_class).permute(0, 3, 1, 2)
        input =   input[:,1:,:,:] # background channel out
        
        target = F.one_hot(target, num_classes=n_class).permute(0, 3, 1, 2)
        target = target[:,1:,:,:]
    elif loss_type == "BCE":
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



def log_creator(path_str):
    log_filename = datetime.datetime.now().strftime(path_str)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(log_filename)
    logger.addHandler(handler)
    
    return logger



#----------------------------------------------------------------
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor, to_pil_image
import torchvision.ops as ops

def show_box(img, targets, labels, classes, denormal):
    
    if type(img) == 'torch.Tensor':
        print("tensor")
        
    if len(img.shape) == 4:
        print("batch")
        img = img[0,:]
    
    #image = img.clone()
    image = img
    
    if denormal == True:
        image = torch_denormalize(image)
    #img = np.float64(img)
    
    image = to_pil_image(image)
    draw = ImageDraw.Draw(image)
    targets = np.array(targets)
    W, H = image.size
    
    #-- colors
    colors = np.random.randint(0, 255, size=(80,3), dtype='uint8') # bbox color 
    #--

    for tg,label in zip(targets,labels):
        id_ = int(label) # class
        bbox = tg[:4]    # [x1, y1, x2, y2]

        color = [int(c) for c in colors[id_]]
        name = classes[id_]

        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=tuple(color), width=3)
        draw.text((bbox[0], bbox[1]), name, fill=(255,255,255,0))
    
    plt.figure(figsize=(12,12))
    plt.imshow(np.array(image))



#----------------------------------------------------------------
# Object Detection Utils 

def gen_anc_centers(out_size):
    out_h, out_w = out_size
    
    anc_pts_x = torch.arange(0, out_w) + 0.5 
    anc_pts_y = torch.arange(0, out_h) + 0.5 
    
    return anc_pts_x, anc_pts_y


def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    n_anc_boxes = len(anc_scales) * len(anc_ratios)
    anc_base = torch.zeros(1, anc_pts_x.size(dim=0) \
                              , anc_pts_y.size(dim=0), n_anc_boxes, 4) 
    # shape - [1, Hmap, Wmap, n_anchor_boxes, 4]
    
    for ix, xc in enumerate(anc_pts_x):
        for jx, yc in enumerate(anc_pts_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4))
            c = 0
            for i, scale in enumerate(anc_scales):
                for j, ratio in enumerate(anc_ratios):
                    w = scale * ratio
                    h = scale
                    
                    xmin = xc - w / 2
                    ymin = yc - h / 2
                    xmax = xc + w / 2
                    ymax = yc + h / 2

                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                    c += 1

            anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)
            
    return anc_base


def get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all,device=None):
    '''
    # B - Batch Size
    # w_amap - width of the output activation map
    # h_wmap - height of the output activation map
    # n_anc_boxes - number of anchor boxes per an anchor point
    # max_objects - max number of objects in a batch of images
    # anc_boxes_tot - total number of anchor boxes in the image i.e, w_amap * h_amap * n_anc_boxes
    
    '''
    
    # flatten anchor boxes
    anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
    # get total anchor boxes for a single image
    tot_anc_boxes = anc_boxes_flat.size(dim=1)
    
    # create a placeholder to compute IoUs amongst the boxes
    ious_mat = torch.zeros((batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1)))

    # compute IoU of the anc boxes with the gt boxes for all the images
    for i in range(batch_size):
        gt_bboxes = gt_bboxes_all[i]
        anc_boxes = anc_boxes_flat[i]
        #------------------
        if device:
            anc_boxes = anc_boxes.to(device)
            gt_bboxes = gt_bboxes.to(device)
        ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)
        
    return ious_mat


def project_bboxes(bboxes, width_scale_factor, height_scale_factor, mode='a2p'):
    assert mode in ['a2p', 'p2a']
    
    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = (proj_bboxes == -1) # indicating padded bboxes
    
    if mode == 'a2p':
        # activation map to pixel image
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        # pixel image to activation map
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor
        
    proj_bboxes.masked_fill_(invalid_bbox_mask, -1) # fill padded bboxes back with -1
    proj_bboxes.resize_as_(bboxes)
    
    return proj_bboxes

def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping,device):
    pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]
    
    if device:
        gt_cx, gt_cy, gt_w, gt_h = gt_cx.to(device), gt_cy.to(device), gt_w.to(device), gt_h.to(device)
        anc_cx, anc_cy, anc_w, anc_h = anc_cx.to(device), anc_cy.to(device), anc_w.to(device), anc_h.to(device)

    tx_ = (gt_cx - anc_cx)/anc_w
    ty_ = (gt_cy - anc_cy)/anc_h
    tw_ = torch.log(gt_w / anc_w)
    th_ = torch.log(gt_h / anc_h)

    return torch.stack([tx_, ty_, tw_, th_], dim=-1)


def generate_proposals(anchors, offsets,device):

    if device:
        anchors, offsets = anchors.to(device), offsets.to(device)

    # change format of the anchor boxes from 'xyxy' to 'cxcywh'
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

    # apply offsets to anchors to create proposals
    proposals_ = torch.zeros_like(anchors)
    proposals_[:,0] = anchors[:,0] + offsets[:,0]*anchors[:,2]
    proposals_[:,1] = anchors[:,1] + offsets[:,1]*anchors[:,3]
    proposals_[:,2] = anchors[:,2] * torch.exp(offsets[:,2])
    proposals_[:,3] = anchors[:,3] * torch.exp(offsets[:,3])

    # change format of proposals back from 'cxcywh' to 'xyxy'
    proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

    return proposals


def get_req_anchors(anc_boxes_all, gt_bboxes_all, gt_classes_all, pos_thresh=0.7, neg_thresh=0.2, device=None):
    '''
    Prepare necessary data required for training
    
    Input
    ------
    anc_boxes_all - torch.Tensor of shape (B, w_amap, h_amap, n_anchor_boxes, 4)
        all anchor boxes for a batch of images
    gt_bboxes_all - torch.Tensor of shape (B, max_objects, 4)
        padded ground truth boxes for a batch of images
    gt_classes_all - torch.Tensor of shape (B, max_objects)
        padded ground truth classes for a batch of images
        
    Returns
    ---------
    positive_anc_ind -  torch.Tensor of shape (n_pos,)
        flattened positive indices for all the images in the batch
    negative_anc_ind - torch.Tensor of shape (n_pos,)
        flattened positive indices for all the images in the batch
    GT_conf_scores - torch.Tensor of shape (n_pos,), IoU scores of +ve anchors
    GT_offsets -  torch.Tensor of shape (n_pos, 4),
        offsets between +ve anchors and their corresponding ground truth boxes
    GT_class_pos - torch.Tensor of shape (n_pos,)
        mapped classes of +ve anchors
    positive_anc_coords - (n_pos, 4) coords of +ve anchors (for visualization)
    negative_anc_coords - (n_pos, 4) coords of -ve anchors (for visualization)
    positive_anc_ind_sep - list of indices to keep track of +ve anchors
    '''
    # get the size and shape parameters
    B, w_amap, h_amap, A, _ = anc_boxes_all.shape
    N = gt_bboxes_all.shape[1] # max number of groundtruth bboxes in a batch
    
    # get total number of anchor boxes in a single image
    tot_anc_boxes = A * w_amap * h_amap
    
    # get the iou matrix which contains iou of every anchor box
    # against all the groundtruth bboxes in an image
    iou_mat = get_iou_mat(B, anc_boxes_all, gt_bboxes_all, device=device)
    
    # for every groundtruth bbox in an image, find the iou 
    # with the anchor box which it overlaps the most
    max_iou_per_gt_box, _ = iou_mat.max(dim=1, keepdim=True)
    
    # get positive anchor boxes
    
    # condition 1: the anchor box with the max iou for every gt bbox
    # return boolean mask fits the condition
    positive_anc_mask = torch.logical_and(iou_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0) 
    # condition 2: anchor boxes with iou above a threshold with any of the gt bboxes
    positive_anc_mask = torch.logical_or(positive_anc_mask, iou_mat > pos_thresh)
    positive_anc_ind_sep = torch.where(positive_anc_mask)[0] # get separate indices in the batch
    # combine all the batches and get the idxs of the +ve anchor boxes
    positive_anc_mask = positive_anc_mask.flatten(start_dim=0, end_dim=1)
    positive_anc_ind = torch.where(positive_anc_mask)[0]
    
    # for every anchor box, get the iou and the idx of the
    # gt bbox it overlaps with the most
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
    max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)
    
    # get iou scores of the +ve anchor boxes
    GT_conf_scores = max_iou_per_anc[positive_anc_ind]
    
    # get gt classes of the +ve anchor boxes
    
    # expand gt classes to map against every anchor box
    # (8,8) => (8,3249,8)
    gt_classes_expand = gt_classes_all.view(B, 1, N).expand(B, tot_anc_boxes, N)
    # for every anchor box, consider only the class of the gt bbox it overlaps with the most
    if device:
        gt_classes_expand = gt_classes_expand.to(device)
        max_iou_per_anc_ind = max_iou_per_anc_ind.to(device)

    GT_class = torch.gather(gt_classes_expand, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1)
    # combine all the batches and get the mapped classes of the +ve anchor boxes
    GT_class = GT_class.flatten(start_dim=0, end_dim=1)
    GT_class_pos = GT_class[positive_anc_ind]
    
    # get gt bbox coordinates of the +ve anchor boxes
    
    # expand all the gt bboxes to map against every anchor box
    gt_bboxes_expand = gt_bboxes_all.view(B, 1, N, 4).expand(B, tot_anc_boxes, N, 4)
    # for every anchor box, consider only the coordinates of the gt bbox it overlaps with the most
    if device:
        gt_bboxes_expand = gt_bboxes_expand.to(device)

    GT_bboxes = torch.gather(gt_bboxes_expand, -2, max_iou_per_anc_ind.reshape(B, tot_anc_boxes, 1, 1).repeat(1, 1, 1, 4))
    # combine all the batches and get the mapped gt bbox coordinates of the +ve anchor boxes
    GT_bboxes = GT_bboxes.flatten(start_dim=0, end_dim=2)
    GT_bboxes_pos = GT_bboxes[positive_anc_ind]
    
    # get coordinates of +ve anc boxes
    anc_boxes_flat = anc_boxes_all.flatten(start_dim=0, end_dim=-2) # flatten all the anchor boxes
    positive_anc_coords = anc_boxes_flat[positive_anc_ind]
    
    # calculate gt offsets
    GT_offsets = calc_gt_offsets(positive_anc_coords, GT_bboxes_pos, device=device)
    
    # get -ve anchors
    
    # condition: select the anchor boxes with max iou less than the threshold
    negative_anc_mask = (max_iou_per_anc < neg_thresh)
    negative_anc_ind = torch.where(negative_anc_mask)[0]
    # sample -ve samples to match the +ve samples
    negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (positive_anc_ind.shape[0],))]
    negative_anc_coords = anc_boxes_flat[negative_anc_ind]
    
    return positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, \
         positive_anc_coords, negative_anc_coords, positive_anc_ind_sep
         
         
         
#--------------------------------------------- 
# Face Detection Dataset Utils 
from bs4 import BeautifulSoup
import matplotlib.patches as patches


def generate_box(obj):
    
    xmin = float(obj.find('xmin').text)
    ymin = float(obj.find('ymin').text)
    xmax = float(obj.find('xmax').text)
    ymax = float(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]

adjust_label = 1

def generate_label(obj):
    
    '''
    Target Classes belongs to :

    with_mask
    mask_weared_incorrect
    without_mask
    
    Dataset contain images :

    with_mask : 79.37%
    mask_weared_incorrect : 3.02%
    without_mask : 17.61%
    
    As there are 3 classes in target, we can use 3 colors for cascading the face.

    Red --> without_mask
    Green --> with_mask
    Yellow -- > mask_weared_incorrect
    
    
    '''

    if obj.find('name').text == "with_mask":

        return 1 + adjust_label

    elif obj.find('name').text == "mask_weared_incorrect":

        return 2 + adjust_label

    # 바로 return 되는 케이스는 "without_mask" case 이다. 코드상 명시되도록 수정필요. 

    return 0 + adjust_label

def generate_target(file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "html.parser")
        objects = soup.find_all("object")

        num_objs = len(objects)

        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))

        boxes = torch.as_tensor(boxes, dtype=torch.float32) 
        labels = torch.as_tensor(labels, dtype=torch.int64) 
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        return target

def plot_image_from_output(img, annotation):
    
    #annotation = [{k: v.to("cpu") for k, v in t.items()} for t in annotation]
    img = img.clone()
    img = img.permute(1,2,0).detach().cpu()
    
    fig,ax = plt.subplots(1,figsize=(10,8))
    ax.imshow(img)
    
    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 1 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        
        elif annotation['labels'][idx] == 2 :
            
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
            
        else :
        
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')

        ax.add_patch(rect)

    plt.show()
    
    
    
def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i] # predict
        # pred_boxes = output['boxes']
        # pred_scores = output['scores']
        # pred_labels = output['labels']

        true_positives = torch.zeros(output['boxes'].shape[0])   # 예측 객체 개수
 
        annotations = targets[sample_i]  # actual
        target_labels = annotations['labels'] if len(annotations) else []
        if len(annotations):    # len(annotations) = 3
            detected_boxes = []
            target_boxes = annotations['boxes']

            for pred_i, (pred_box, pred_label) in enumerate(zip(output['boxes'], output['labels'])): # 예측값에 대해서..

                # If targets are found break
                if len(detected_boxes) == len(target_labels): # annotations -> target_labels
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)   # box_index : 실제 어떤 바운딩 박스랑 IoU 가 가장 높은지 index
                if iou >= iou_threshold and box_index not in detected_boxes: # iou만 맞으면 통과?
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]  # 예측된거랑 실제랑 매핑해서 하나씩 index 채움
        batch_metrics.append([true_positives, output['scores'], output['labels']])
    return batch_metrics




def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = torch.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = torch.unique(target_cls)   # 2가 거의 예측안됨

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = torch.cumsum(1 - tp[i],-1)
            tpc = torch.cumsum(tp[i],-1)

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = torch.tensor(np.array(p)), torch.tensor(np.array(r)), torch.tensor(np.array(ap))
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap





#---

def draw_mask(image, mask_generated) :
    
    '''
    ref : https://inside-machinelearning.com/en/plot-segmentation-mask/
    
    image shape : (573, 763, 3)
    mask shape  : (573, 763, 3)
    
    '''
    masked_image = image.copy()
    
    if len(mask_generated.shape) == 2:
        mask_generated = np.stack((mask_generated,mask_generated,mask_generated),axis=-1)

    masked_image = np.where(mask_generated.astype(int),
                            np.array([0,255,0], dtype='uint8'),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)


def draw_masks_fromList(image, masks_generated, labels, colors) :
    
    '''
    # image 
    => (h , w , c)
    
    # masks_generated
    
    single mask shape
     =>([ number of obejcts , 823, 1171])
    masks_generated = [i for i in qa]
    
    # LABELS = [[0], [0]] # 0 for bird, 1 for
     
    # COLORS = [(255,0,0),(255,255,0)] 
    
    '''
    masked_image = image.copy()
    for i in range(len(masks_generated)) :
        picked_color =colors[((i+1) % len(colors))]
        print(picked_color)
        masked_image = np.where(np.repeat(masks_generated[i][:, :, np.newaxis], 3, axis=2),
                                #-- color iteration n ignore label 
                                np.asarray( picked_color, dtype='uint8'),
                                
                                #np.asarray(colors[int(labels[  (len(colors)) % (i+1) ][-1])], dtype='uint8'),
                                
                                #-- fix one color 
                                #(255,0,0),
                                masked_image)                                                                                             
        masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)



#----------------------------------------------
# Dota Visualization

def read_txt(file_path):
    with open(file_path, "r",encoding='utf-8',errors='ignore') as file:
        lines = file.readlines()

    return lines


#----------------------------------------------    
# Box Converter 

def box_converter(pred_box, input_type, output_type):
    
    pred_box = pred_box.detach().cpu()
    pred_box = np.array(pred_box)
    
    if (input_type == "4_coord_center") & (output_type == "4_coord_minmax"):
        
        
        x_center, y_center = pred_box[0],pred_box[1]
        width, height = pred_box[2],pred_box[3]
        angle = pred_box[4]

        x_min = x_center - width/2
        y_min = y_center - height/2

        x_max = x_center + width/2
        y_max = y_center + height/2
            
        box = [x_min,y_min,x_max,y_max,angle]
        
    
    def four_to_eight(box):
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]
        
        box = [x_min, y_min, x_min,y_max, x_max, y_min, x_max, y_max]       
        return box 
    
    #box = four_to_eight(box)
    
    return box
         
def box_rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    
    modified from answer here: https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)


def rotated_to_rectangle(rotated_boxes):

    rectangle_boxes = []
    for box in rotated_boxes:
        x_min = min([i[0] for i in box])
        y_min = min([i[1] for i in box])
        
        x_max = max([i[0] for i in box])
        y_max = max([i[1] for i in box])
        
        poly_box = [[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min]]
        rectangle_boxes.append(poly_box)   
    
    return rectangle_boxes

# --- affine-transform and poly

# 0 convertinf to poly type
def convert_to_polygon(box):
    x, y, width, height = box[0:4]

    # Calculate the coordinates of the four corners
    x1, y1 = x-0.5*width, y-0.5*height
    x2, y2 = x-0.5*width, y+0.5*height
    x3, y3 = x+0.5*width, y+0.5*height
    x4, y4 = x+0.5*width, y-0.5*height

    # Create a list of points representing the polygon
    polygon = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    return polygon




#----------------------------------------------------
# Metric 

# Classification 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

class Metric_Classification:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def regression_metrics(self,labels_, predictions_):
        mae = nn.L1Loss()
        mse = nn.MSELoss()
        mse_losses = mse(labels_,predictions_)
        mae_losses = mae(labels_,predictions_)
        
        return mse_losses, mae_losses
        
    def classification_metrics(self, labels_, predictions_):
        precision, recall, f1, _ = precision_recall_fscore_support(labels_, predictions_, average='weighted')
        accuracy = accuracy_score(labels_, predictions_)
        
        return precision, recall, f1, accuracy 
    

class Metric_Classification_Valid:
    '''
    Classification Class to Singleton Pattern
    
    '''

    __shared_state = {"precision" :[],
                      "recall" :[],
                      "f1" :[],
                      "accuracy":[]}
    def __init__(self):
        self.__dict__ = self.__shared_state

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def regression_metrics(self,labels_, predictions_):
        mae = nn.L1Loss()
        mse = nn.MSELoss()
        mse_losses = mse(labels_,predictions_)
        mae_losses = mae(labels_,predictions_)
        
        return mse_losses, mae_losses
        
    def classification_metrics(self, labels_, predictions_):
        precision, recall, f1, _ = precision_recall_fscore_support(labels_, predictions_, average='weighted')
        accuracy = accuracy_score(labels_, predictions_)
        
        self.__dict__["precision"].append(precision)
        self.__dict__["recall"].append(recall)
        self.__dict__["f1"].append(f1)
        self.__dict__["accuracy"].append(accuracy)
        

        #return precision, recall, f1, accuracy 


class Metric_Classification_Train:
    '''
    Classification Class to Singleton Pattern
    
    '''

    __shared_state = {"precision" :[],
                      "recall" :[],
                      "f1" :[],
                      "accuracy":[]}
    def __init__(self):
        self.__dict__ = self.__shared_state


    def regression_metrics(self,labels_, predictions_):
        mae = nn.L1Loss()
        mse = nn.MSELoss()
        mse_losses = mse(labels_,predictions_)
        mae_losses = mae(labels_,predictions_)
        
        return mse_losses, mae_losses
        
    def classification_metrics(self, labels_, predictions_):
        precision, recall, f1, _ = precision_recall_fscore_support(labels_, predictions_, average='weighted')
        accuracy = accuracy_score(labels_, predictions_)
        
        self.__dict__["precision"].append(precision)
        self.__dict__["recall"].append(recall)
        self.__dict__["f1"].append(f1)
        self.__dict__["accuracy"].append(accuracy)


#--- 
import threading

class Singleton:
    _instance = None
    _lock = threading.Lock()
    _data = {
        "a" :[]
    }

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


#-- class lazy for SingleTon 
# class LazyInstantiation:
class Metric_Train:
    _instance = None
    _lock = threading.Lock()
    _shared_state = {"precision" :[],
                      "recall" :[],
                      "f1" :[],
                      "accuracy":[]}
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Metric_Train, cls).__new__(cls)
        return cls._instance
 
    @classmethod
    def getInstance(cls):
        if not cls._instance:
            cls._instance = Metric_Train()
        return cls._instance
 
    def data_update(self,v):
        self._shared_state["a"] += v

    def eval_classification(self, labels_, predictions_):
        precision, recall, f1, _ = precision_recall_fscore_support(labels_, predictions_, average='weighted')
        accuracy = accuracy_score(labels_, predictions_)
        
        self._shared_state["precision"].append(precision)
        self._shared_state["recall"].append(recall)
        self._shared_state["f1"].append(f1)
        self._shared_state["accuracy"].append(accuracy)

        return precision, recall, f1, accuracy

class Metric_Valid:
    _instance = None
    _lock = threading.Lock()
    _shared_state = {"precision" :[],
                      "recall" :[],
                      "f1" :[],
                      "accuracy":[]}
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Metric_Valid, cls).__new__(cls)
        return cls._instance
 
    @classmethod
    def getInstance(cls):
        if not cls._instance:
            cls._instance = Metric_Valid()
        return cls._instance
 
    def data_update(self,v):
        self._shared_state["a"] += v

    def eval_classification(self, labels_, predictions_):
        precision, recall, f1, _ = precision_recall_fscore_support(labels_, predictions_, average='weighted')
        accuracy = accuracy_score(labels_, predictions_)
        
        self._shared_state["precision"].append(precision)
        self._shared_state["recall"].append(recall)
        self._shared_state["f1"].append(f1)
        self._shared_state["accuracy"].append(accuracy)

        return precision, recall, f1, accuracy

#--- Label Distribution for pdf 

def pdf_fn(x):
  '''
  x to pdf distribution 
  '''
  x_pdf = torch.exp( -(x)**2 /2  ) * 1/( torch.pi * torch.sqrt(torch.tensor(2)) )
  return x_pdf


def label_to_dist(label):
  '''
  - int label [0,1,0,0,0]
  - distrbuted label = [1,0,1,2,3]
  int label will be distributed by a interval "gap_" variable below 
  need to update gap_=> 0.05 interval
 
  '''

  gap_ = 1
  target_label_index = torch.where(label==1)[0][0].detach().cpu().numpy()
  label_dist = [i for i in np.arange(target_label_index,0,-gap_)] + [i for i in np.arange(0,len(label) - target_label_index,gap_ )]
  return label_dist




def label_to_dist_torch(label):
  '''
  patch note 
  label_to_dist => label_to_dist_torch
  make label to dist function can work with torch 
  
  - int label [0,1,0,0,0]
  - distrbuted label = [1,0,1,2,3]
  int label will be distributed by a interval "gap_" variable below 
  need to update gap_=> 0.05 interval
 
  '''

  gap_ = 1
  #target_label_index = torch.where(label == 1)[0][0]
  #label_dist = list(range(target_label_index, 0, -gap_)) + list(range(0, len(label) - target_label_index, gap_))
  target_label_index = torch.where(label==1)[0][0]
  label_dist = [i for i in np.arange(target_label_index,0,-gap_)] + [i for i in np.arange(0,len(label) - target_label_index,gap_ )]
  label_dist = torch.tensor(label_dist)
  
  
  return label_dist