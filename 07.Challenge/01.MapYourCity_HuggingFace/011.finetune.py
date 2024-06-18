#--
import sys
sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace")
import map_dataset
import map_train
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

#--- argparser
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace/configs/finetune.yaml')
args = parser.parse_args()
cfg = argparse.Namespace(**yaml.load(open(args.cfg), Loader=yaml.SafeLoader))

#--- loggers 
log_root = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace/logs"
log_name = f"{cfg.RUN_VERSION}_{cfg.MODEL}" 
tb_logger = TensorBoardLogger(root_dir=log_root,name=log_name)
csv_logger = CSVLogger(root_dir=log_root,name=log_name)
logger = RS_utils.log_creator(os.path.join(log_root, log_name))

#--- Online logger
wandb.init(project='MapYourCity')
# 실행 이름 설정
wandb.run.name = f'{cfg.RUN_VERSION}_{cfg.MODEL}'
wandb.run.save()


#--- init fabric
if cfg.FABRIC: 
    fabric = Fabric(accelerator="cuda", devices=cfg.DEVICES, strategy="ddp",loggers=[tb_logger, csv_logger])
    fabric.launch()
    
    #--- metric code 
    # use torchmetrics instead of manually computing the accuracy
    test_acc = Accuracy(task="multiclass", num_classes=cfg.CLASSES_NUM).to(fabric.device)

#--- Data 
input_path = "/mnt/hdd/eric/.tmp_ipy/00.Data/Map_Your_City/building-age-dataset/"
train_path = input_path + "train/data/"
test_path = input_path + "test/data/"
train_df = pd.read_csv(input_path + "train/train-set.csv")
test_df = pd.read_csv(input_path + "test/test-set.csv") 

#--- data split 
names_data = os.listdir(train_path)
if cfg.SAMPLE:
    parse_idx = int(len(names_data) * 0.01)
    names_data = names_data[:parse_idx]
names_train, names_valid = train_test_split(names_data, test_size=0.1, random_state=1)

#-----------------------------------------------------------------------------------------
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
#-----


#--- train and valid selection need
train_set = map_dataset.Map_Dataset_v1(names_train,train_path,max_size=data_config['input_size'][1]) 
valid_set = map_dataset.Map_Dataset_v1(names_valid,train_path,max_size=data_config['input_size'][1])  

#--- dataloader 
trainloader = DataLoader(train_set, cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
testloader = DataLoader(valid_set, cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, pin_memory=True) 

train_loss_fn = LabelSmoothCrossEntropy(smoothing=0.1)
val_loss_fn = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), cfg.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
scheduler = StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
scaler = GradScaler(enabled=cfg.AMP)

print("fabric")
#--- fabric
if cfg.FABRIC:
    model, optimizer = fabric.setup(model, optimizer)
    trainloader,testloader = fabric.setup_dataloaders(trainloader,testloader)

model.train()
iteration = 0
#--- score var
best_top1_acc, best_top5_acc = 0.0, 0.0

for epoch in range(cfg.EPOCHS):
    
    accum_time = 0
    # Train 
    for iteration, (X, y) in enumerate(trainloader):
        
        iter_start = time.time()
        
        #with autocast(enabled=cfg.AMP):    
        pred = model(X)
        loss = val_loss_fn(pred, y)

        optimizer.zero_grad()
        fabric.backward(scaler.scale(loss))
        scaler.step(optimizer)
        scaler.update()
        
        
        iter_end =time.time()
        iter_time = iter_end - iter_start
        accum_time += iter_time
        
        time_total = len(trainloader) * iter_time 
        time_remaining = time_total - accum_time
        
        log_dict=   {
            "epoch" : epoch,
            "iteration" : iteration,
            "progress" : float(str(iteration / len(trainloader))[0:4]),
            "time_remaining" :  time.strftime("%H:%M:%S", time.gmtime(time_remaining)),
            "time_total" : time.strftime("%H:%M:%S", time.gmtime(time_total)) ,
            "loss" : float(str(loss.item())[0:6]),  
            "batch_size" : int(cfg.BATCH_SIZE),
            "lr" : optimizer.param_groups[0]['lr']
        }
        
        logger.info(log_dict)
        wandb.log(log_dict)
        

    # Valid
    top1_acc, top5_acc = map_train.test_v1(testloader, model, val_loss_fn)
    if top1_acc > best_top1_acc:
        best_top1_acc = top1_acc
        best_top5_acc = top5_acc
        torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, f"{cfg.MODEL}_epoch_{epoch}.pth"))
        print(f" Best Top-1 Accuracy: [red]{(best_top1_acc):>0.1f}%[/red]\tBest Top-5 Accuracy: [red]{(best_top5_acc):>0.1f}%[/red]\n")
    
    log_dict_test = {
    "epoch" : epoch,
    "batch_size" : int(cfg.BATCH_SIZE),
    "lr" : optimizer.param_groups[0]['lr'],
    "best_top1_acc" : top1_acc,
    "best_top5_acc" : top5_acc
    }
    logger.info(log_dict_test)
    wandb.log(log_dict_test)
    


# #---
# # TESTING LOOP
# model.eval()
# test_loss = 0
# with torch.no_grad():
#     for data, target in test_loader:
#         # NOTE: no need to call `.to(device)` on the data, target
#         output = model(data)
#         test_loss += F.nll_loss(output, target, reduction="sum").item()

#         # WITHOUT TorchMetrics
#         # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#         # correct += pred.eq(target.view_as(pred)).sum().item()

#         # WITH TorchMetrics
#         test_acc(output, target)

#         if hparams.dry_run:
#             break

# # all_gather is used to aggregated the value across processes
# test_loss = fabric.all_gather(test_loss).sum() / len(test_loader.dataset)

# print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({100 * test_acc.compute():.0f}%)\n")
# test_acc.reset()
# #---


