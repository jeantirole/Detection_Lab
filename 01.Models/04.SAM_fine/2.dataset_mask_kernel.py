# patchfiy & label handling

from tqdm import tqdm
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

ISAID_CLASSES = ('background', 'ship', 'store_tank', 'baseball_diamond',
               'tennis_court', 'basketball_court', 'Ground_Track_Field',
               'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
               'Harbor')

palette = ISAID_PALETTE = {
    0: (0, 0, 0), 1: (0, 0, 63), 2: (0, 63, 63), 3: (0, 63, 0), 4: (0, 63, 127),
    5: (0, 63, 191), 6: (0, 63, 255), 7: (0, 127, 63), 8: (0, 127, 127),
    9: (0, 0, 127), 10: (0, 0, 191), 11: (0, 0, 255), 12: (0, 191, 127),
    13: (0, 127, 191), 14: (0, 127, 255), 15: (0, 100, 155)}

def __mask_encoding__(label):
    
    zero_label = np.zeros(label.shape[:2], dtype=np.uint8)
    
    for k,v in palette.items():
        zero_label[np.all(label == v, axis=-1)] = k 
    
    return zero_label


masks_cate = []

root_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/05.1024_masks"
mask_list = sorted(os.listdir(root_path))

mask_list = mask_list[:int(len(mask_list)*1)]

print(mask_list[0:4])


cnt = 0
for mask in mask_list:
    
    mask = os.path.join(root_path, mask)
    mask = cv2.imread(mask, cv2.IMREAD_COLOR)
    mask =cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    ww = __mask_encoding__(mask)
    masks_cate.append(ww)
    
    cnt+=1
    print("progress : ", cnt, "total : ", len(mask_list) )
    
masks_cate = np.array(masks_cate, dtype=object)
np.save("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/01.Models/04.SAM_fine/0.data/06.1024_masks_categorized/mask_processed_all.npy", masks_cate)

