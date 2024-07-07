# # Ensemble Train
# - 3 data (street view, top view, sentinel2)
# - 3data x 5fold = 15 models for ensemble




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
import random
import matplotlib.pyplot as plt
from rich.console import Console
from sklearn.model_selection import StratifiedKFold
from wandb.integration.lightning.fabric import WandbLogger

#--- all infos
inference_dict ={
    'models':[],
    'folds' :[],
    'data' :[],
    'cfgs':[],
    'predictions':[],
    'labels':[]
}

#--- argparser
cfgs_names = ['finetune_21.yaml', 'finetune_20.yaml','finetune_22.yaml']
for cfg_name in cfgs_names:    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=os.path.join('/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace/configs', cfg_name))
    args = parser.parse_args(args=[])
    cfg = argparse.Namespace(**yaml.load(open(args.cfg), Loader=yaml.SafeLoader))
    
    for fold_ in range(cfg.N_SPLIT):        
        inference_dict['cfgs'].append(cfg)
        inference_dict['folds'].append(fold_)
        inference_dict['data'].append(cfg.DATA_TYPE)
        print("Model run version : ", cfg.RUN_VERSION)
        print("Model run fold : ", fold_)
        print("Data type : ", cfg.DATA_TYPE)



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

#--- delete
cfg.SAMPLE = False
#cfg.SAMPLE_PERCENT = 0.01
#--- delete

if cfg.SAMPLE:
    parse_idx = int(len(names_data) * cfg.SAMPLE_PERCENT)
    names_data = names_data[:parse_idx]
    names_label = names_label[:parse_idx]
    
print(names_data[0:4])


for cfg in inference_dict['cfgs']:   
    model = timm.create_model(
    cfg.MODEL,
    pretrained=True,
    num_classes=cfg.CLASSES_NUM )

    #--- data config and transform
    data_config = timm.data.resolve_model_data_config(model)
    data_transform = timm.data.create_transform(**data_config, is_training=False)

    inference_dict['models'].append(model)
    print("#------------------------------------")
    print(" Model Name : ",cfg.MODEL)


#--- all the candidates for ensembles based on validation score 

saved_root = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace/output"
check_points = sorted(os.listdir(saved_root))
target_runs = list(set([ cfg.RUN_VERSION for cfg in inference_dict['cfgs']]))

print("split ns : ", cfg.N_SPLIT)


def find_best_model(target_run_n):
    
    target_runs_0 = [i for i in check_points if str(i.split("_")[0]) == str(target_run_n) ]
    
    best_model_runs = []
    for fold_n in range(0,cfg.N_SPLIT):
        fold_s = [ i for i in target_runs_0 if str(i.split("_")[-5]) == str(fold_n) ]
        #print(fold_s)
        best_model = ""
        best_score = 0
        for fq in fold_s:
            score =  float(fq.split("_")[-3])
            if score > best_score:
                score = best_score
                best_model = fq
        best_model_runs.append(best_model)
    return best_model_runs  


#--- find all 
global_best_models = []
for tg in target_runs:
    fold_best_models = find_best_model(tg)
    global_best_models.extend(fold_best_models)


# Define the categories in the desired order
categories = ['streetview', 'topview', 'sentinel2']

# Create a dictionary to hold lists of file paths for each category
categorized_files = {category: [] for category in categories}

# Categorize the file paths
for path in global_best_models:
    for category in categories:
        if category in path:
            categorized_files[category].append(path)
            break

# Reorder the file paths based on the desired order
ordered_file_paths = []
for category in categories:
    ordered_file_paths.extend(categorized_files[category])

# Print the ordered file paths
for path in ordered_file_paths:
    print(path)


ckpt_paths =[ os.path.join(saved_root,i) for i in ordered_file_paths]
for i,model in enumerate(inference_dict['models']):
    model.load_state_dict(torch.load(ckpt_paths[i])['model'] )


class EnsembleModel(torch.nn.Module):

    '''
    - two models will be ensembled

    '''

    def __init__(self,  model_0_fold_0, model_0_fold_1, model_0_fold_2, model_0_fold_3, model_0_fold_4,
                        model_1_fold_0, model_1_fold_1, model_1_fold_2, model_1_fold_3, model_1_fold_4,
                        model_2_fold_0, model_2_fold_1, model_2_fold_2, model_2_fold_3, model_2_fold_4, 
                        ):
        super().__init__()
        
        #--- model 0 
        self.model_0_fold_0 = model_0_fold_0
        self.model_0_fold_1 = model_0_fold_1
        self.model_0_fold_2 = model_0_fold_2
        self.model_0_fold_3 = model_0_fold_3
        self.model_0_fold_4 = model_0_fold_4

        #--- model 1 
        self.model_1_fold_0 = model_1_fold_0 
        self.model_1_fold_1 = model_1_fold_1 
        self.model_1_fold_2 = model_1_fold_2 
        self.model_1_fold_3 = model_1_fold_3 
        self.model_1_fold_4 = model_1_fold_4

        #--- model 2 
        self.model_2_fold_0 = model_2_fold_0 
        self.model_2_fold_1 = model_2_fold_1 
        self.model_2_fold_2 = model_2_fold_2 
        self.model_2_fold_3 = model_2_fold_3 
        self.model_2_fold_4 = model_2_fold_4 

        self.classifier = torch.nn.Linear(7 * 15, 7)
        
    def forward(self, x1, x2, x3):
        x1_0 = self.model_0_fold_0(x1)
        x1_1 = self.model_0_fold_1(x1)
        x1_2 = self.model_0_fold_2(x1)
        x1_3 = self.model_0_fold_3(x1)
        x1_4 = self.model_0_fold_4(x1)

        x2_0 = self.model_1_fold_0(x2)
        x2_1 = self.model_1_fold_1(x2)
        x2_2 = self.model_1_fold_2(x2)
        x2_3 = self.model_1_fold_3(x2)
        x2_4 = self.model_1_fold_4(x2)

        x3_0 = self.model_2_fold_0(x3)
        x3_1 = self.model_2_fold_1(x3)
        x3_2 = self.model_2_fold_2(x3)
        x3_3 = self.model_2_fold_3(x3)
        x3_4 = self.model_2_fold_4(x3)

        out = torch.cat( (x1_0, x1_1,x1_2,x1_3,x1_4,
                          x2_0, x2_1,x2_2,x2_3,x2_4,
                          x3_0, x3_1,x3_2,x3_3,x3_4), dim=1)
        out = self.classifier(out)
        return out
    


# Example usage
# Assuming each model is defined and instantiated as model1, model2, ..., model15
models = inference_dict['models']
ensemble_model = EnsembleModel(inference_dict['models'][0],inference_dict['models'][1],inference_dict['models'][2],inference_dict['models'][3],inference_dict['models'][4],
                               inference_dict['models'][5],inference_dict['models'][6],inference_dict['models'][7],inference_dict['models'][8],inference_dict['models'][9],
                               inference_dict['models'][10],inference_dict['models'][11],inference_dict['models'][12],inference_dict['models'][13],inference_dict['models'][14] )


for param in ensemble_model.parameters():
    param.requires_grad = False

for param in ensemble_model.classifier.parameters():
    param.requires_grad = True    

model = ensemble_model



if cfg.LOSS_FN =="MSE":
    train_loss_fn = torch.nn.MSELoss()
    val_loss_fn = torch.nn.MSELoss()
if cfg.LOSS_FN =="MAE":
    train_loss_fn = torch.nn.L1Loss()
    val_loss_fn = torch.nn.L1Loss()
elif cfg.LOSS_FN =="CE":     
    train_loss_fn = CrossEntropyLoss() 
    val_loss_fn = CrossEntropyLoss()


#optimizer = AdamW(model.parameters(), cfg.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)


#--- Metric
#metric_obj = RS_utils.Metric_Classification()



#-- kfold 
SKF = StratifiedKFold(n_splits=cfg.N_SPLIT, shuffle=True, random_state=cfg.RANDOM_STATE)


for fold, (train_ids, valid_ids) in enumerate(SKF.split(names_data,names_label)):
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    #--- loggers 
    log_root = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace/logs"
    model1 = inference_dict['cfgs'][0].RUN_VERSION
    model2 = inference_dict['cfgs'][5].RUN_VERSION
    model3 = inference_dict['cfgs'][10].RUN_VERSION

    log_name = f'Ensemble_Model_v2_{model1}_{model2}_{model3}'
    logger = RS_utils.log_creator(os.path.join(log_root, log_name))
    wandb_run_name = f'Ensemble_{model1}_{model2}_{model3}_{cfg.LOSS_FN}_{cfg.MODEL[0:10]}_nfold_{cfg.N_SPLIT}'
    GROUP_NAME = "experiment-" + wandb.util.generate_id()
    
    #wandb.init(project="MapYourCity", group=GROUP_NAME, name=wandb_run_name, job_type=wandb_run_name, config={"fold": fold})
    logger_wb = WandbLogger(project="MapYourCity",name=wandb_run_name,group=GROUP_NAME)



    #------------- model 
    
    class EnsembleModel(torch.nn.Module):

        '''
        - two models will be ensembled

        '''

        def __init__(self,  model_0_fold_0, model_0_fold_1, model_0_fold_2, model_0_fold_3, model_0_fold_4,
                            model_1_fold_0, model_1_fold_1, model_1_fold_2, model_1_fold_3, model_1_fold_4,
                            model_2_fold_0, model_2_fold_1, model_2_fold_2, model_2_fold_3, model_2_fold_4, 
                            ):
            super().__init__()
            
            #--- model 0 
            self.model_0_fold_0 = model_0_fold_0
            self.model_0_fold_1 = model_0_fold_1
            self.model_0_fold_2 = model_0_fold_2
            self.model_0_fold_3 = model_0_fold_3
            self.model_0_fold_4 = model_0_fold_4

            #--- model 1 
            self.model_1_fold_0 = model_1_fold_0 
            self.model_1_fold_1 = model_1_fold_1 
            self.model_1_fold_2 = model_1_fold_2 
            self.model_1_fold_3 = model_1_fold_3 
            self.model_1_fold_4 = model_1_fold_4

            #--- model 2 
            self.model_2_fold_0 = model_2_fold_0 
            self.model_2_fold_1 = model_2_fold_1 
            self.model_2_fold_2 = model_2_fold_2 
            self.model_2_fold_3 = model_2_fold_3 
            self.model_2_fold_4 = model_2_fold_4 

            self.classifier = torch.nn.Linear(7 * 15, 7)
            
        def forward(self, x1, x2, x3):
            x1_0 = self.model_0_fold_0(x1)
            x1_1 = self.model_0_fold_1(x1)
            x1_2 = self.model_0_fold_2(x1)
            x1_3 = self.model_0_fold_3(x1)
            x1_4 = self.model_0_fold_4(x1)

            x2_0 = self.model_1_fold_0(x2)
            x2_1 = self.model_1_fold_1(x2)
            x2_2 = self.model_1_fold_2(x2)
            x2_3 = self.model_1_fold_3(x2)
            x2_4 = self.model_1_fold_4(x2)

            x3_0 = self.model_2_fold_0(x3)
            x3_1 = self.model_2_fold_1(x3)
            x3_2 = self.model_2_fold_2(x3)
            x3_3 = self.model_2_fold_3(x3)
            x3_4 = self.model_2_fold_4(x3)

            out = torch.cat( (x1_0, x1_1,x1_2,x1_3,x1_4,
                            x2_0, x2_1,x2_2,x2_3,x2_4,
                            x3_0, x3_1,x3_2,x3_3,x3_4), dim=1)
            out = self.classifier(out)
            return out
        


    # Example usage
    # Assuming each model is defined and instantiated as model1, model2, ..., model15
    models = inference_dict['models']
    ensemble_model = EnsembleModel(inference_dict['models'][0],inference_dict['models'][1],inference_dict['models'][2],inference_dict['models'][3],inference_dict['models'][4],
                                inference_dict['models'][5],inference_dict['models'][6],inference_dict['models'][7],inference_dict['models'][8],inference_dict['models'][9],
                                inference_dict['models'][10],inference_dict['models'][11],inference_dict['models'][12],inference_dict['models'][13],inference_dict['models'][14] )


    for param in ensemble_model.parameters():
        param.requires_grad = False

    for param in ensemble_model.classifier.parameters():
        param.requires_grad = True    

    model = ensemble_model

    #--- data config and transform
    data_config = timm.data.resolve_model_data_config(inference_dict['models'][0])
    data_transform = timm.data.create_transform(**data_config, is_training=False)

    logger.info(data_config)
    logger.info(data_transform)
    #-----

    #--- init fabric
    if cfg.FABRIC: 
        fabric = Fabric(accelerator="cuda", devices=cfg.DEVICES, strategy="ddp",loggers=[logger_wb])
        fabric.launch()
    
    #---    
    train_names = [names_data[i] for i in train_ids]
    valid_names = [names_data[i] for i in valid_ids]
    
    #--- train and valid selection need
    train_set = map_dataset.Map_Dataset_v12(train_names,train_path,max_size=data_config['input_size'][1],cfg=cfg,split='train') 
    valid_set = map_dataset.Map_Dataset_v12(valid_names,train_path,max_size=data_config['input_size'][1],cfg=cfg,split='valid')  

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
        train_loss_fn = CrossEntropyLoss() 
        val_loss_fn = CrossEntropyLoss()
        
        #train_loss_fn = LabelSmoothCrossEntropy()
        #val_loss_fn = LabelSmoothCrossEntropy()
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
    #model.train()
    #--- score var

    for epoch in range(cfg.EPOCHS):
        
        accum_time = 0
        metric_obj_train = RS_utils.Metric_Train()
        # Train 
        for iteration, (str_imgs,top_imgs, sentinel_imgs, labels,dist_labels) in enumerate(trainloader):
            
            iter_start = time.time()
            
            # this is ensemble for three images
            pred = model(str_imgs, top_imgs, sentinel_imgs)
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

        #--- Metric
        metric_obj_valid = RS_utils.Metric_Valid()

        #---
        if cfg.LOSS_FN == "CE" or "KL":
            #-------------------------------------------------
            #-- evaluate validation only for zero rank 
            precision_,recall_, f1_, acc_  = map_train.test_for_ensemble_v1(trainloader, model, val_loss_fn,cfg,metric_obj_train)
            valid_precision, valid_recall, valid_f1, valid_accuracy = map_train.test_for_ensemble_v1(testloader, model, val_loss_fn,cfg,metric_obj_valid)
            
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
            f'Ensemble_{model1}_{model2}_{model3}_{cfg.LOSS_FN}_{cfg.MODEL[0:10]}_nfold_{cfg.N_SPLIT}'
            fabric.save(os.path.join(cfg.SAVE_DIR, f"Ensemble_{model1}_{model2}_{model3}_{cfg.LOSS_FN}_{cfg.MODEL[0:10]}_nfold_{cfg.N_SPLIT}_fold_{fold}_recall_{save_recall}_epoch_{epoch}.pth"), state)
            #torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, f"{cfg.RUN_VERSION}_{cfg.MODEL}_{cfg.DATA_TYPE}_{cfg.LOSS_FN}_f1_{valid_f1}_epoch_{epoch}.pth"))
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()