
import torch
from torch.cuda.amp import GradScaler, autocast
from rich.console import Console
from utils.losses import LabelSmoothCrossEntropy, CrossEntropyLoss
from utils.utils import fix_seeds, setup_cudnn, create_progress_bar
from utils.metrics import compute_accuracy




def train(dataloader, model, loss_fn, optimizer, scheduler, scaler, device, epoch, cfg,logger,fabric):

    
    model.train()
    progress = create_progress_bar()
    lr = scheduler.get_last_lr()[0]
    task_id = progress.add_task(description="", total=len(dataloader), epoch=epoch+1, epochs=cfg.EPOCHS, lr=lr, loss=0.)

    iteration = 0
    
    for X, y in dataloader:
        #X, y = X.to(device), y.to(device)

        with autocast(enabled=cfg.AMP):    
            pred = model(X)
            loss = loss_fn(pred, y)

        optimizer.zero_grad()
        fabric.backward(scaler.scale(loss))
        scaler.step(optimizer)
        scaler.update()
        
        log_dict=   {
            "epoch" : epoch,
            "iteration" : iteration,
            "loss" : "{:.5f}".format(loss.item()),  
            "batch_size" : int(cfg.BATCH_SIZE),
            "lr" : optimizer.param_groups[0]['lr']
        }        
        logger.info(log_dict)
        iteration += 1

        progress.update(task_id, description="", advance=1, refresh=True, loss=loss.item())

    scheduler.step()
    progress.stop()


def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, top1_acc, top5_acc = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            acc1, acc5 = compute_accuracy(pred, y, topk=(1, 5))
            top1_acc += acc1 * X.shape[0]
            top5_acc += acc5 * X.shape[0]

    test_loss /= num_batches
    top1_acc /= size
    top5_acc /= size
    
    return top1_acc, top5_acc


def test_v1(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, top1_acc, top5_acc = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            acc1, acc5 = compute_accuracy(pred, y, topk=(1, 5))
            top1_acc += acc1 * X.shape[0]
            top5_acc += acc5 * X.shape[0]

    test_loss /= num_batches
    top1_acc /= size
    top5_acc /= size
    
    return top1_acc, top5_acc
