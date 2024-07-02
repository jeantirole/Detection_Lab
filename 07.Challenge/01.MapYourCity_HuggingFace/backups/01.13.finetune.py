
'''
patch note 01.13
1. Refactoring for Fabric and WanDB   

patch note 01.12
1. Ensemble(Street View,Top View) with KL loss  

patch note 01.11
1. Street View learning with KL loss  

patch note 01.10
1. Distributed Train for "Label deviation" 

patch note 01.9
1. Label => Distribution
"label_deviation" hyperparam added
2. Parallel train for KL div 

patch note 01.8 

1. Label => Label pdf distribution 
2. KL Div loss added  

patch note 01.7 

1. orthophoto.tif piepes added 
2. Dataloader => gets street img, top-view img 

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
parser.add_argument('--cfg', type=str, \
                    default='/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace/configs/finetune_test.yaml')
args = parser.parse_args()
cfg = argparse.Namespace(**yaml.load(open(args.cfg), Loader=yaml.SafeLoader))

#--- loggers 
log_root = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace/logs"
log_name = f"{cfg.RUN_VERSION}_{cfg.MODEL}" 
logger = RS_utils.log_creator(os.path.join(log_root, log_name))
wandb_run_name = f'{cfg.RUN_VERSION}_{cfg.DATA_TYPE}_{cfg.MODEL[0:10]}'
logger_wb = WandbLogger(project="MapYourCity",name=wandb_run_name)


# logger.log_hyperparams({
#     "lr": cfg.LR,
#     "batch_size": cfg.BATCH_SIZE,
#     "num_epochs": cfg.EPOCHS,
# })


#--- Online logger
#wandb.run.name = f'{cfg.RUN_VERSION}_{cfg.DATA_TYPE}_{cfg.MODEL}'
#wandb.run.save()


#--- init fabric
if cfg.FABRIC: 
    fabric = Fabric(accelerator="cuda", devices=cfg.DEVICES, strategy="ddp",loggers=[logger_wb])
    fabric.launch()
    
#--- Data 
input_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/Map_Your_City/building-age-dataset/"
train_path = input_path + "train/data/"
test_path = input_path + "test/data/"
train_df = pd.read_csv(input_path + "train/train-set.csv")
test_df = pd.read_csv(input_path + "test/test-set.csv") 

#--- data split 
names_data = os.listdir(train_path)
if cfg.SAMPLE:
    parse_idx = int(len(names_data) * 0.05)
    names_data = names_data[:parse_idx]
names_train, names_valid = train_test_split(names_data, test_size=0.1, random_state=1)

#-----------------------------------------------------------------------------------------
#--- model

if cfg.TIMM: 
    model = timm.create_model(
    cfg.MODEL,
    pretrained=True,
    num_classes=cfg.CLASSES_NUM ) 

    #--- data config and transform
    data_config = timm.data.resolve_model_data_config(model)
    data_transform = timm.data.create_transform(**data_config, is_training=False)

    logger.info(data_config)
    logger.info(data_transform)
    #-----
else:
    model = eval(cfg.MODEL)(cfg.VARIANT, cfg.PRETRAINED, cfg.CLASSES_NUM, cfg.IMAGE_SIZE) 


#--- train and valid selection need
if cfg.TIMM:
    train_set = map_dataset.Map_Dataset_v9(names_train,train_path,max_size=data_config['input_size'][1],cfg=cfg,split='train') 
    valid_set = map_dataset.Map_Dataset_v9(names_valid,train_path,max_size=data_config['input_size'][1],cfg=cfg,split='valid')  


#--- dataloader 
trainloader = DataLoader(train_set, cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
testloader = DataLoader(valid_set, cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, pin_memory=True) 

#train_loss_fn = LabelSmoothCrossEntropy(smoothing=0.1)

if cfg.LOSS_FN =="MSE":
    train_loss_fn = torch.nn.MSELoss()
    val_loss_fn = torch.nn.MSELoss()
elif cfg.LOSS_FN =="MAE":
    train_loss_fn = torch.nn.L1Loss()
    val_loss_fn = torch.nn.L1Loss()
elif cfg.LOSS_FN =="CE":     
    train_loss_fn = CrossEntropyLoss() 
    val_loss_fn = CrossEntropyLoss()
elif cfg.LOSS_FN =="KL":
    train_loss_fn = torch.nn.KLDivLoss(reduction='batchmean') 
    val_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')


optimizer = AdamW(model.parameters(), cfg.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
#scheduler = StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
#scaler   = GradScaler(enabled=cfg.AMP)



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
    for iteration, (str_imgs,top_imgs, _, labels,dist_labels) in enumerate(trainloader):
        
        iter_start = time.time()
        
        imgs = str_imgs
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
        #wandb.log(log_dict)
        # print("#---- test !!")
        # print(len(trainloader))
        # print(len(trainloader)*0.1)
        # print( (iteration+1) % (len(trainloader)*0.1)  )
        #--- Evaluation for train
        if  progress!= 0 and (progress*100) % 10 == 0:
            logger.info("#------------------ Evaluation for train")
            predictions = torch.argmax(pred, -1).cpu()
            if cfg.LOSS_FN == "KL":
                dist_labels = dist_labels.detach().cpu()
                labels = dist_labels
            labels = labels.cpu()
            # calculate 
            precision_,recall_, f1_, acc_ = metric_obj_train.eval_classification(labels,predictions)
            #print("fabric.global_rank : ",fabric.global_rank, "precision : ", precision_)
            
            precision_result = fabric.all_reduce(precision_, reduce_op="sum")
            recall_result = fabric.all_reduce(recall_, reduce_op="sum")
            f1_result = fabric.all_reduce(f1_, reduce_op="sum")
            acc_result = fabric.all_reduce(acc_, reduce_op="sum")

            #-- ddp barrier 
            #print("fabric.global_rank : ",fabric.global_rank)
            #fabric.barrier()
            #--
            if fabric.global_rank ==0:

                log_dict_train = {
                "epoch" : epoch,
                "iteration" : iteration,
                "progress" : float(str(iteration / len(trainloader))[0:4]),
                "batch_size" : int(cfg.BATCH_SIZE),
                "lr" : optimizer.param_groups[0]['lr'],
                "train precision": precision_result / len(cfg.DEVICES),
                "train recall": recall_result / len(cfg.DEVICES),
                "train f1" : f1_result / len(cfg.DEVICES),
                "train accuracy" : acc_result / len(cfg.DEVICES),
                }
                logger.info(log_dict_train)
                fabric.log_dict(log_dict_train)

    #---
    #--- Metric
    metric_obj_valid = RS_utils.Metric_Valid()

    #---
    if cfg.LOSS_FN == "CE" or "KL":
        #-------------------------------------------------
        #-- evaluate validation only for zero rank 
        valid_precision, valid_recall, valid_f1, valid_accuracy = map_train.test_v6(testloader, model, val_loss_fn,cfg,metric_obj_valid)
        
        #---- fabric barrier for evaluation synchronization
        precision_result_valid = fabric.all_reduce(valid_precision, reduce_op="sum").item()
        recall_result_valid = fabric.all_reduce(valid_recall, reduce_op="sum").item()
        f1_result_valid = fabric.all_reduce(valid_f1, reduce_op="sum").item()
        acc_result_valid = fabric.all_reduce(valid_accuracy, reduce_op="sum").item()

        
        if fabric.global_rank ==0:

            log_dict_test = {
            "epoch" : epoch,
            "batch_size" : int(cfg.BATCH_SIZE),
            "lr" : optimizer.param_groups[0]['lr'],
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
        }
        
        save_recall = recall_result_valid / len(cfg.DEVICES)
        save_recall = str(save_recall)[0:6]
        # Instead of `torch.save(...)`
        fabric.save(os.path.join(cfg.SAVE_DIR, f"{cfg.RUN_VERSION}_{cfg.MODEL}_{cfg.DATA_TYPE}_{cfg.LOSS_FN}_recall_{save_recall}_epoch_{epoch}.pth"), state)
        #torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, f"{cfg.RUN_VERSION}_{cfg.MODEL}_{cfg.DATA_TYPE}_{cfg.LOSS_FN}_f1_{valid_f1}_epoch_{epoch}.pth"))
        
