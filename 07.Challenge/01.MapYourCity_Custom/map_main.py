import imageio.v2 as io 
import matplotlib.pyplot as plt         
import pandas as pd
import rasterio 
import random
from tqdm import tqdm                         
import os          
import shutil
from PIL import Image
import numpy as np
import cv2
      
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#--- torch
import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import transforms, datasets

#--- module
import sys 
sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_Custom")
from map_model import get_activation, get_normalization, SE_Block 
from map_model import CoreUnet 

sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Libs")
import RS_dataset
import RS_models
import RS_utils


#--- header
EXEC_VER = 0
MAINFOLDER = '/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity/MapYourCity'                         
BATCH_SIZE = 16     
EPOCHS = 999
SEED = 2 # For the random seed         
#SEED = random.randint(1, 10000)
torch.cuda.empty_cache()
random.seed(SEED)
np.random.seed(SEED) 
torch.manual_seed(SEED)   
torch.cuda.manual_seed_all(SEED)
print('The random seed is: ' + str(SEED) + '.')  

#-- logger 
log_path = f'/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_Custom/exps/ver_{EXEC_VER}_%Y-%m-%d_%H-%M-%S.log'
logger = RS_utils.log_creator(log_path)

#--- path
input_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/Map_Your_City/building-age-dataset/"
train_path = input_path + "train/data/"
test_path = input_path + "test/data/"

#--- device
device = torch.device("cuda:0")
print('Device: ' + str(device))   

#--- Load csv files
train_df = pd.read_csv(input_path + "train/train-set.csv")
test_df = pd.read_csv(input_path + "test/test-set.csv") 

#--- data split 
names_data = os.listdir(train_path)
names_train, names_valid = train_test_split(names_data, test_size=0.2, random_state=1)

#--- dataset class 
class Map_Dataset(torch.utils.data.Dataset):      
    def __init__(self, list_IDs): 
        self.list_IDs = list_IDs
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index): 
        ID = self.list_IDs[index] 
        X1 = cv2.imread(train_path + ID + '/street.jpg')
        X1 = cv2.resize(X1, (256, 256)) 
        
        X2 = cv2.imread(train_path + ID + '/orthophoto.tif')  
        X2 = cv2.resize(X2, (256, 256)) 
        
        X3 = rasterio.open(train_path + ID + '/s2_l2a.tif').read() 
        X3 = np.transpose(X3, [1, 2, 0]) 
        
        y = int(open(train_path + ID + '/label.txt', "r").read())
        return X1, X2, X3, y 
    

#--- train and valid selection need
train_set = Map_Dataset(names_train) 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  

valid_set = Map_Dataset(names_valid)   
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)


#--- model 
from map_model import MyResNet18,CoreUnet

#--- resnet combined
model1 = torchvision.models.resnet152(pretrained=True)
model2 = torchvision.models.resnet152(pretrained=True)
yourresnet18 = MyResNet18(model1,model2) 

#--- corenet 
CHANNELS = 12
core_model = CoreUnet(
input_dim=CHANNELS,
output_dim=1)

class Custom_Model(nn.Module):
    def __init__(self, res_model,core_model):
        super(Custom_Model, self).__init__()  # Initialize nn.Module
        self.l0 = nn.Linear(4480, 7) # 7 classes
        self.res_model = res_model
        self.core_model = core_model

    def forward(self, x1, x2, x3):
        batch, _, _, _ = x1.shape
        
        x1 = self.res_model.features(x1)
        x2 = self.res_model.features2(x2)
        x3 = self.core_model(x3) 
        
        x1 = F.adaptive_avg_pool2d(x1, 1).reshape(batch, -1)    
        x2 = F.adaptive_avg_pool2d(x2, 1).reshape(batch, -1) 
        x3 = F.adaptive_avg_pool2d(x3, 1).reshape(batch, -1) 
        
        x1 = torch.cat((x1, x2, x3), 1)   
        l0 = self.l0(x1)
        
        return l0 
    
#--- exec
model = Custom_Model(yourresnet18, core_model)

# we also use: http://github.com/ESA-PhiLab/AI4EO-Challenge-Building-Sustainability          
model = model.to(device)
model.train()  
criterion = nn.CrossEntropyLoss()           
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.5e-3)  

from map_valid import validate

best_test_bpd = 0         
howoften = 4
iteration = 0

#--- train 
for epoch in range(1,4):  # # loop over the dataset multiple times
   print("Epoch:", epoch)  
   for idx, batch in tqdm(enumerate(train_loader)):
        
        pixel_values, pixel_values2, pixel_values3, labels = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32), batch[2].to(device, dtype=torch.float32), batch[3].to(device) 
        pixel_values = pixel_values.permute(0, 3, 1, 2)        
        pixel_values2 = pixel_values2.permute(0, 3, 1, 2) 
        pixel_values3 = pixel_values3.permute(0, 3, 1, 2)
        
        optimizer.zero_grad()
        
        outputs = model(pixel_values, pixel_values2, pixel_values3)  
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #--- log dict 
        log_dict=   {
            "epoch" : epoch,
            "iteration" : iteration,
            "iteration_by_epoch" : (epoch // len(train_loader)) // BATCH_SIZE,
            "loss" : "{:.5f}".format(loss.item()),  
            "batch_size" : int(BATCH_SIZE),
            "lr" : optimizer.param_groups[0]['lr']
        }        
        logger.info(log_dict)
        iteration += 1
        #---
        
        if idx % 10 == 0:
          print("iter:", f"{idx} / {len(train_loader)}","Loss:", loss.item())  

        #--- valid test and save
        if (epoch % howoften == 0) and (idx == 0):
          accToCheck = validate(model)   
          if accToCheck > best_test_bpd:     
              best_test_bpd = accToCheck  
              torch.save(model.state_dict(), f'./exps/ver_{EXEC_VER}_epoch_{epoch}_iteration_{iteration}_modelB.pt')     
              
              # script save
              RS_utils.save_current_file(EXEC_VER)

torch.save(model.state_dict(), './exps/last_model.pt')   