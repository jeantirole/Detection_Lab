import numpy as np 
import matplotlib.pyplot as plt
#from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np 

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

    if obj.find('name').text == "with_mask":

        return 1 + adjust_label

    elif obj.find('name').text == "mask_weared_incorrect":

        return 2 + adjust_label

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
    
    img = img.cpu().permute(1,2,0)
    
    fig,ax = plt.subplots(1)
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