
'''
patch note 01.22
label smoothing 


patch note 01.22
sentinel2 
64x64 => 224x224 
train pipes changed 


path note 01.20
holdout => k-fold ! 


'''


#--
import sys
sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace")
import map_dataset
import map_train
from models import *

sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Libs")
import RS_dataset
import RS_models
import RS_utils
#--- torch
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
#--- loss functions
from utils.losses import LabelSmoothCrossEntropy, CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
#---
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from torchmetrics.classification import Accuracy
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import argparse
import yaml 
import timm
import numpy as np 
import time
import wandb
from rich.console import Console
from wandb.integration.lightning.fabric import WandbLogger



#--- argparser
parser = argparse.ArgumentParser()
config_root = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace/configs"
yaml_ = "finetune_27.yaml"
parser.add_argument('--cfg', type=str, \
                    default=os.path.join(config_root,yaml_))
args = parser.parse_args()
cfg = argparse.Namespace(**yaml.load(open(args.cfg), Loader=yaml.SafeLoader))

#--- Data 
input_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/Map_Your_City/building-age-dataset/"
train_path = input_path + "train/data/"
test_path = input_path + "test/data/"
train_df = pd.read_csv(input_path + "train/train-set.csv")
test_df = pd.read_csv(input_path + "test/test-set.csv") 

#--- data split 
names_data = sorted( os.listdir(train_path) )

names_label = []
for ID in names_data:
    y = int(open(train_path + ID + '/label.txt', "r").read())
    names_label.append(y)

if cfg.SAMPLE:
    parse_idx = int(len(names_data) * cfg.SAMPLE_PERCENT)
    names_data = names_data[:parse_idx]
    names_label = names_label[:parse_idx]


#-- kfold 
SKF = StratifiedKFold(n_splits=cfg.N_SPLIT, shuffle=True, random_state=cfg.RANDOM_STATE)


for fold, (train_ids, valid_ids) in enumerate(SKF.split(names_data,names_label)):
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    #--- loggers 
    log_root = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace/logs"
    log_name = f"{cfg.RUN_VERSION}_{cfg.MODEL}" 
    logger = RS_utils.log_creator(os.path.join(log_root, log_name))
    wandb_run_name = f'{cfg.RUN_VERSION}_{cfg.DATA_TYPE}_{cfg.LOSS_FN}_{cfg.MODEL[0:10]}_nfold_{cfg.N_SPLIT}'
    GROUP_NAME = "experiment-" + wandb.util.generate_id()
    #wandb.init(project="MapYourCity", group=GROUP_NAME, name=wandb_run_name, job_type=wandb_run_name, config={"fold": fold})
    logger_wb = WandbLogger(project="MapYourCity",name=wandb_run_name,group=GROUP_NAME)

    #--- model
    model = timm.create_model(
    cfg.MODEL,
    pretrained=True,
    num_classes=cfg.CLASSES_NUM ) 

    #--- data config and transform
    data_config = timm.data.resolve_model_data_config(model)
    data_transform = timm.data.create_transform(**data_config, is_training=False)

    logger.info(data_config)
    logger.info(data_transform)
    logger.info(cfg)
    #-----

    #--- init fabric
    if cfg.FABRIC: 
        fabric = Fabric(accelerator="cuda", devices=cfg.DEVICES, strategy="ddp",loggers=[logger_wb])
        fabric.launch()
    
    if fabric.global_rank ==0:
        logger_wb.experiment.config.update(cfg)

    
    #---    
    train_names = [names_data[i] for i in train_ids]
    valid_names = [names_data[i] for i in valid_ids]
    
    #--- train and valid selection need
    train_set = map_dataset.Map_Dataset_v14(train_names,train_path,max_size=data_config['input_size'][1],cfg=cfg,split='train') 
    valid_set = map_dataset.Map_Dataset_v14(valid_names,train_path,max_size=data_config['input_size'][1],cfg=cfg,split='valid')  

    #--- dataloader 
    trainloader = DataLoader(train_set, cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    testloader = DataLoader(valid_set, cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, pin_memory=True) 

    if cfg.LOSS_FN =="MSE":
        train_loss_fn = torch.nn.MSELoss()
        val_loss_fn = torch.nn.MSELoss()
    elif cfg.LOSS_FN =="MAE":
        train_loss_fn = torch.nn.L1Loss()
        val_loss_fn = torch.nn.L1Loss()
    elif cfg.LOSS_FN =="CE":     
        if cfg.LABEL_SMOOTHING == False:
            train_loss_fn = CrossEntropyLoss() 
            val_loss_fn = CrossEntropyLoss()
            
        elif cfg.LABEL_SMOOTHING == True:
            train_loss_fn = LabelSmoothCrossEntropy()
            val_loss_fn = LabelSmoothCrossEntropy()
    elif cfg.LOSS_FN =="KL":
        train_loss_fn = torch.nn.KLDivLoss(reduction='batchmean') 
        val_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')


    if cfg.OPTIMIZER == "AdamW":
        optimizer = AdamW(model.parameters(), cfg.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    #optimizer = AdamW(model.parameters(), cfg.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    #scheduler = StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    #scaler   = GradScaler(enabled=cfg.AMP)
    if cfg.SCHEDULER == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.LEARNING_RATE_CYCLE, T_mult=1, eta_min=0)

    print("fabric")
    #--- fabric
    if cfg.FABRIC:
        model, optimizer = fabric.setup(model, optimizer)
        trainloader,testloader = fabric.setup_dataloaders(trainloader,testloader)
        
    else:
        model = model.to(cfg.DEVICE)
        
    #logger_wb.watch(model)
    model.train()
    #--- score var

    for epoch in range(cfg.EPOCHS):
        
        accum_time = 0
        metric_obj_train = RS_utils.Metric_Train()
        # Train 
        for iteration, (str_imgs,top_imgs, sentinel_imgs, labels,dist_labels) in enumerate(trainloader):
            
            iter_start = time.time()
            
            if cfg.DATA_TYPE == "streetview":
                imgs = str_imgs
            elif cfg.DATA_TYPE == "topview":
                imgs = top_imgs
            elif cfg.DATA_TYPE =="sentinel2":
                imgs = sentinel_imgs
            
            if not cfg.FABRIC:
                imgs = imgs.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)
            #with autocast(enabled=cfg.AMP):    
            
            pred = model(imgs)
            optimizer.zero_grad()

            if cfg.LOSS_FN == "MSE" or cfg.LOSS_FN == "MAE" :
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)           
                labels = labels.to(torch.float32)
                loss = train_loss_fn(pred, labels)
            elif cfg.LOSS_FN =="CE":
                loss = train_loss_fn(pred, labels)
            elif cfg.LOSS_FN =="KL":
                # pred
                pred = torch.nn.functional.softmax(pred, dim=1)
                loss = train_loss_fn(pred.log(), labels)

            if cfg.FABRIC:
                fabric.backward(loss)
                optimizer.step()
            else:
                loss.backward()
                optimizer.step()        
            
            iter_end =time.time()
            iter_time = iter_end - iter_start
            accum_time += iter_time
            
            time_total = len(trainloader) * iter_time 
            time_remaining = time_total - accum_time
            
            progress = float(str(iteration / len(trainloader))[0:4])

            if fabric.global_rank ==0:
                log_dict=   {
                    "fold" :fold, 
                    "epoch" : epoch,
                    "iteration" : iteration,
                    "progress" : progress,
                    "time_remaining" :  time.strftime("%H:%M:%S", time.gmtime(time_remaining)),
                    "time_total" : time.strftime("%H:%M:%S", time.gmtime(time_total)) ,
                    "loss" : float(str(loss.item())[0:6]),  
                    "batch_size" : int(cfg.BATCH_SIZE),
                    "lr" : optimizer.param_groups[0]['lr']
                }
                
                logger.info(log_dict)
                fabric.log_dict(log_dict)

        #--- Scheduler Step LR
        scheduler.step() 

        #--- Metric
        metric_obj_valid = RS_utils.Metric_Valid()

        #---
        if cfg.LOSS_FN == "CE" or "KL":
            #-------------------------------------------------
            #-- evaluate validation only for zero rank 
            precision_,recall_, f1_, acc_  = map_train.test_v7(trainloader, model, val_loss_fn,cfg,metric_obj_train)
            valid_precision, valid_recall, valid_f1, valid_accuracy = map_train.test_v7(testloader, model, val_loss_fn,cfg,metric_obj_valid)
            
            #---- train eval 
            precision_result = fabric.all_reduce(precision_, reduce_op="sum").item()
            recall_result = fabric.all_reduce(recall_, reduce_op="sum").item()
            f1_result = fabric.all_reduce(f1_, reduce_op="sum").item()
            acc_result = fabric.all_reduce(acc_, reduce_op="sum").item()

            #---- valid eval 
            precision_result_valid = fabric.all_reduce(valid_precision, reduce_op="sum").item()
            recall_result_valid = fabric.all_reduce(valid_recall, reduce_op="sum").item()
            f1_result_valid = fabric.all_reduce(valid_f1, reduce_op="sum").item()
            acc_result_valid = fabric.all_reduce(valid_accuracy, reduce_op="sum").item()
            
            if fabric.global_rank ==0:

                log_dict_test = {
                "fold" :fold,
                "epoch" : epoch,
                "batch_size" : int(cfg.BATCH_SIZE),
                "lr" : optimizer.param_groups[0]['lr'],
                "train precision": precision_result / len(cfg.DEVICES),
                "train recall": recall_result / len(cfg.DEVICES),
                "train f1" : f1_result / len(cfg.DEVICES),
                "train accuracy" : acc_result / len(cfg.DEVICES),
                "valid_precision" : precision_result_valid / len(cfg.DEVICES),
                "valid_recall" : recall_result_valid / len(cfg.DEVICES),
                "valid_f1" : f1_result_valid / len(cfg.DEVICES),
                "valid_accuracy" : acc_result_valid / len(cfg.DEVICES)}
                logger.info(log_dict_test)
                #wandb.log(log_dict_test)
                fabric.log_dict(log_dict_test)

        elif cfg.LOSS_FN == "MSE" or cfg.LOSS_FN == "MAE":
            
            #-- evaluate validation
            regression_loss = map_train.test_v4_3(testloader, model, val_loss_fn,cfg,metric_obj)
            regression_loss = str(regression_loss)[0:6]

            log_dict_test = {
            "epoch" : epoch,
            "batch_size" : int(cfg.BATCH_SIZE),
            "lr" : optimizer.param_groups[0]['lr'],
            "loss type":cfg.LOSS_FN,
            "valid loss" :regression_loss}
            logger.info(log_dict_test)
            #wandb.log(log_dict_test)
            fabric.log_dict(log_dict_test)

        
        #-- model save
        if cfg.LOSS_FN=="MSE" or cfg.LOSS_FN=="MAE":
            torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, f"{cfg.RUN_VERSION}_{cfg.MODEL}_{cfg.LOSS_FN}_{regression_loss}_epoch_{epoch}.pth")) 
        else:
            # Define the state of your program/loop
            state = {
                "model": model,
                "optimizer": optimizer,
                "epoch": epoch,
                "fold":fold,
            }
            
            save_recall = recall_result_valid / len(cfg.DEVICES)
            save_recall = str(save_recall)[0:6]
            # Instead of `torch.save(...)`
            if cfg.MODEL_SAVE == True:
                fabric.save(os.path.join(cfg.SAVE_DIR, f"{cfg.RUN_VERSION}_{cfg.MODEL}_{cfg.DATA_TYPE}_{cfg.LOSS_FN}_fold_{fold}_recall_{save_recall}_epoch_{epoch}.pth"), state)
                #torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, f"{cfg.RUN_VERSION}_{cfg.MODEL}_{cfg.DATA_TYPE}_{cfg.LOSS_FN}_f1_{valid_f1}_epoch_{epoch}.pth"))
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
torch.cuda.empty_cache()