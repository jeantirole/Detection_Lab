
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


def test_v1(dataloader, model, loss_fn,cfg):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, top1_acc, top5_acc = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            
            if not cfg.FABRIC:
                X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            acc1, acc5 = compute_accuracy(pred, y, topk=(1, 5))
            top1_acc += acc1 * X.shape[0]
            top5_acc += acc5 * X.shape[0]

    test_loss /= num_batches
    top1_acc /= size
    top5_acc /= size
    
    return top1_acc, top5_acc


def test_v2(dataloader, model, loss_fn,cfg, metric_object):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, top1_acc, top5_acc = 0, 0, 0

    predictions_ = []
    labels_ = []
    with torch.no_grad():
        for X, y in dataloader:
            
            if not cfg.FABRIC:
                X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
            pred = model(X)
            
            #--
            predictions = torch.argmax(pred,-1)
            predictions_.extend(predictions.cpu())
            labels_.extend(y.cpu())
            #--
            test_loss += loss_fn(pred, y).item()
            
    #---    
    precision, recall, f1, accuracy  = metric_object.classification_metrics(labels_,predictions_)
    
    
    return precision, recall, f1, accuracy


#--------------------------------
def test_v3(dataloader, model, loss_fn,cfg, metric_object):
    '''
    CE and Regression losses split 
    '''
    if cfg.LOSS_FN == "CE":
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, top1_acc, top5_acc = 0, 0, 0

        predictions_ = []
        labels_ = []
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)
                
                #--
                predictions = torch.argmax(pred,-1)
                predictions_.extend(predictions.cpu())
                labels_.extend(y.cpu())
                #--
                test_loss += loss_fn(pred, y).item()
                
        #---    
        precision, recall, f1, accuracy  = metric_object.classification_metrics(labels_,predictions_)
        
        
        return precision, recall, f1, accuracy
    
    else:    
        model.eval()
        size = len(dataloader.dataset)
        losses = 0
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)

                #-- 
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                losses += (loss)
                #--
        return losses / len(dataloader)     
    
    
def test_v4(dataloader, model, loss_fn,cfg, metric_object):
    '''
    v4
    dataloader => 2 images (street and topview)
    
     
    '''
    if cfg.LOSS_FN == "CE":
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, top1_acc, top5_acc = 0, 0, 0

        predictions_ = []
        labels_ = []
        with torch.no_grad():
            for str_img, topview_img, y in dataloader:
                
                if not cfg.FABRIC:
                    str_img, topview_img,y = str_img.to(cfg.DEVICE), topview_img.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(topview_img)
                
                #--
                predictions = torch.argmax(pred,-1)
                predictions_.extend(predictions.cpu())
                labels_.extend(y.cpu())
                #--
                test_loss += loss_fn(pred, y).item()
                
        #---    
        precision, recall, f1, accuracy  = metric_object.classification_metrics(labels_,predictions_)
        
        
        return precision, recall, f1, accuracy
    
    else:    
        model.eval()
        size = len(dataloader.dataset)
        losses = 0
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)

                #-- 
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                losses += (loss)
                #--
        return losses / len(dataloader)       
    
def test_v4_1(dataloader, model, loss_fn,cfg, metric_object):
    '''
    v4_1
    dataloader => 3 images (street ,topview1, topview2)
    but use only topview1
    
    v4
    dataloader => 2 images (street and topview)
    
     
    '''
    if cfg.LOSS_FN == "CE" or "KL":
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, top1_acc, top5_acc = 0, 0, 0

        predictions_ = []
        labels_ = []
        with torch.no_grad():
            for str_img, topview_img, _, y in dataloader:
                
                if not cfg.FABRIC:
                    str_img, topview_img,y = str_img.to(cfg.DEVICE), topview_img.to(cfg.DEVICE), y.to(cfg.DEVICE)
                # caution : model only load topview 
                pred = model(topview_img)
                
                #--
                predictions = torch.argmax(pred,-1)
                predictions_.extend(predictions.cpu())
                labels_.extend(y.cpu())
                #--
                #test_loss += loss_fn(pred, y).item()
                
        #---    
        precision, recall, f1, accuracy  = metric_object.classification_metrics(labels_,predictions_)
        
        
        return precision, recall, f1, accuracy
    
    else:    
        model.eval()
        size = len(dataloader.dataset)
        losses = 0
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)

                #-- 
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                losses += (loss)
                #--
        return losses / len(dataloader)       
    


def test_v4_2(dataloader, model, loss_fn,cfg, metric_object):
    '''
    v4_2
    dataloader => 3 images 2 labels ()
    3 images : str_img, topview_img, _, 
    2 labels : y, original_labels
    
    
    v4_1
    dataloader => 3 images (street ,topview1, topview2)
    but use only topview1
    
    v4
    dataloader => 2 images (street and topview)
    
     
    '''
    if cfg.LOSS_FN == "CE" or "KL":
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, top1_acc, top5_acc = 0, 0, 0

        predictions_ = []
        labels_ = []
        print("#-------------- Start Validation")
        with torch.no_grad():
            for str_img, topview_img, _, y in dataloader:
                
                if not cfg.FABRIC:
                    str_img, topview_img,y = str_img.to(cfg.DEVICE), topview_img.to(cfg.DEVICE), y.to(cfg.DEVICE)
                # caution : model only load topview 
                pred = model(topview_img)
                
                #--
                predictions = torch.argmax(pred,-1)
                predictions_.extend(predictions.cpu())
                labels_.extend(y.cpu())
                #--
                #test_loss += loss_fn(pred, y).item()
                
        #---    
        precision, recall, f1, accuracy  = metric_object.classification_metrics(labels_,predictions_)
        
        
        return precision, recall, f1, accuracy
    
    else:    
        model.eval()
        size = len(dataloader.dataset)
        losses = 0
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)

                #-- 
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                losses += (loss)
                #--
        return losses / len(dataloader)       
    
    


def test_v4_3(dataloader, model, loss_fn,cfg, metric_object):
    '''
    v4_3
    topview & steetview validation 
    if cfg.DATA_TYPE=="topview":
        pred = model(topview_img)
    elif cfg.DATA_TYPE =="streetview":
        pred = model(str_img)

    
    v4_2
    dataloader => 3 images 2 labels ()
    3 images : str_img, topview_img, _, 
    2 labels : y, original_labels
    
    
    v4_1
    dataloader => 3 images (street ,topview1, topview2)
    but use only topview1
    
    v4
    dataloader => 2 images (street and topview)
    
     
    '''
    if cfg.LOSS_FN == "CE" or "KL":
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, top1_acc, top5_acc = 0, 0, 0

        predictions_ = []
        labels_ = []
        print("#-------------- Start Validation")

        with torch.no_grad():
            for str_img, topview_img, _, y in dataloader:
                
                if not cfg.FABRIC:
                    str_img, topview_img,y = str_img.to(cfg.DEVICE), topview_img.to(cfg.DEVICE), y.to(cfg.DEVICE)
                # caution : model only load topview 
                if cfg.DATA_TYPE=="topview":
                    pred = model(topview_img)
                elif cfg.DATA_TYPE =="streetview":
                    pred = model(str_img)
                elif cfg.DATA_TYPE =="ensemble":
                    pred = model(str_img, topview_img)

                #--
                predictions = torch.argmax(pred,-1)
                predictions_.extend(predictions.cpu())
                labels_.extend(y.cpu())
                #--
                    
            #--- Singleton
            metric_object.eval_classification(labels_,predictions_)
            
            
    else:    
        model.eval()
        size = len(dataloader.dataset)
        losses = 0
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)

                #-- 
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                losses += (loss)
                #--
        return losses / len(dataloader)       




def test_v4_4(dataloader, model, loss_fn,cfg, metric_object):
    '''
    v4_4
    metric object is created by SingleTone Class


    v4_3
    topview & steetview validation 
    if cfg.DATA_TYPE=="topview":
        pred = model(topview_img)
    elif cfg.DATA_TYPE =="streetview":
        pred = model(str_img)

    
    v4_2
    dataloader => 3 images 2 labels ()
    3 images : str_img, topview_img, _, 
    2 labels : y, original_labels
    
    
    v4_1
    dataloader => 3 images (street ,topview1, topview2)
    but use only topview1
    
    v4
    dataloader => 2 images (street and topview)
    
     
    '''
    if cfg.LOSS_FN == "CE" or "KL":
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, top1_acc, top5_acc = 0, 0, 0

        predictions_ = []
        labels_ = []
        print("#-------------- Start Validation")

        with torch.no_grad():
            for str_img, topview_img, _, y in dataloader:
                
                if not cfg.FABRIC:
                    str_img, topview_img,y = str_img.to(cfg.DEVICE), topview_img.to(cfg.DEVICE), y.to(cfg.DEVICE)
                # caution : model only load topview 
                if cfg.DATA_TYPE=="topview":
                    pred = model(topview_img)
                elif cfg.DATA_TYPE =="streetview":
                    pred = model(str_img)
                elif cfg.DATA_TYPE =="ensemble":
                    pred = model(str_img, topview_img)

                #--
                predictions = torch.argmax(pred,-1)
                predictions_.extend(predictions.cpu())
                labels_.extend(y.cpu())
                #--
                    
            #--- Singleton
            metric_object.classification_metrics(labels_,predictions_)
            
            
    else:    
        model.eval()
        size = len(dataloader.dataset)
        losses = 0
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)

                #-- 
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                losses += (loss)
                #--
        return losses / len(dataloader)       
    
    
def test_v5(dataloader, model, loss_fn,cfg, metric_object):
    '''
    v5
    dataloader => 3 images (street ,topview1, topview2)

    v4
    dataloader => 2 images (street and topview)
    
     
    '''
    if cfg.LOSS_FN == "CE" or "KL":
        print("#--- Loss : ", cfg.LOSS_FN)
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, top1_acc, top5_acc = 0, 0, 0

        predictions_ = []
        labels_ = []
        with torch.no_grad():
            for str_img, topview_img, _, y in dataloader:
                
                if not cfg.FABRIC:
                    str_img, topview_img,y = str_img.to(cfg.DEVICE), topview_img.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(str_img, topview_img)
                
                #--
                predictions = torch.argmax(pred,-1)
                predictions_.extend(predictions.cpu())
                labels_.extend(y.cpu())
                #--
                test_loss += loss_fn(pred, y).item()
                
        #---    
        precision, recall, f1, accuracy  = metric_object.classification_metrics(labels_,predictions_)
        
        
        return precision, recall, f1, accuracy
    
    else:    
        model.eval()
        size = len(dataloader.dataset)
        losses = 0
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)

                #-- 
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                losses += (loss)
                #--
        return losses / len(dataloader)      
    


#---

def test_v6(dataloader, model, loss_fn,cfg, metric_object):
    '''
    v6
    in ddp(fabric) setting,     
    
    v4_3
    topview & steetview validation 
    if cfg.DATA_TYPE=="topview":
        pred = model(topview_img)
    elif cfg.DATA_TYPE =="streetview":
        pred = model(str_img)

    
    v4_2
    dataloader => 3 images 2 labels ()
    3 images : str_img, topview_img, _, 
    2 labels : y, original_labels
    
    
    v4_1
    dataloader => 3 images (street ,topview1, topview2)
    but use only topview1
    
    v4
    dataloader => 2 images (street and topview)
    
     
    '''
    if cfg.LOSS_FN == "CE" or "KL":
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, top1_acc, top5_acc = 0, 0, 0

        predictions_ = []
        labels_ = []
        print("#-------------- Start Validation")

        with torch.no_grad():
            for str_img, topview_img, _, _, y in dataloader:
                
                if not cfg.FABRIC:
                    str_img, topview_img, y = str_img.to(cfg.DEVICE), topview_img.to(cfg.DEVICE), y.to(cfg.DEVICE)
                # caution : model only load topview 
                if cfg.DATA_TYPE=="topview":
                    pred = model(topview_img)
                elif cfg.DATA_TYPE =="streetview":
                    pred = model(str_img)
                elif cfg.DATA_TYPE =="ensemble":
                    pred = model(str_img, topview_img)

                #--
                predictions = torch.argmax(pred,-1)
                predictions_.extend(predictions.cpu())
                labels_.extend(y.cpu())
                #--
                    
        #--- return eval
        pred_, recall_, f1_, acc_ = metric_object.eval_classification(labels_,predictions_)
        return pred_, recall_, f1_, acc_
            
            
    else:    
        model.eval()
        size = len(dataloader.dataset)
        losses = 0
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)

                #-- 
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                losses += (loss)
                #--
        return losses / len(dataloader)     


#000

def test_v7(dataloader, model, loss_fn,cfg, metric_object):
    '''
    v7
    sentienlview added  
    
    
    v6
    in ddp(fabric) setting,     
    
    v4_3
    topview & steetview validation 
    if cfg.DATA_TYPE=="topview":
        pred = model(topview_img)
    elif cfg.DATA_TYPE =="streetview":
        pred = model(str_img)

    
    v4_2
    dataloader => 3 images 2 labels ()
    3 images : str_img, topview_img, _, 
    2 labels : y, original_labels
    
    
    v4_1
    dataloader => 3 images (street ,topview1, topview2)
    but use only topview1
    
    v4
    dataloader => 2 images (street and topview)
    
     
    '''
    if cfg.LOSS_FN == "CE" or "KL":
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, top1_acc, top5_acc = 0, 0, 0

        predictions_ = []
        labels_ = []
        print("#-------------- Start Validation")

        with torch.no_grad():
            for str_img, topview_img, sentinel_imgs, _, y in dataloader:
                
                if not cfg.FABRIC:
                    str_img, topview_img, sentinel_imgs,y = str_img.to(cfg.DEVICE), topview_img.to(cfg.DEVICE), sentinel_imgs.to(cfg.DEVICE),y.to(cfg.DEVICE)
                # caution : model only load topview 
                if cfg.DATA_TYPE=="topview":
                    pred = model(topview_img)
                elif cfg.DATA_TYPE =="streetview":
                    pred = model(str_img)
                elif cfg.DATA_TYPE =="sentinel2":
                    pred = model(sentinel_imgs)
                    
                elif cfg.DATA_TYPE =="ensemble":
                    pred = model(str_img, topview_img)
                    
                elif cfg.DATA_TYPE =="alldata":
                    pred = model(str_img, topview_img, sentinel_imgs)
                #--
                predictions = torch.argmax(pred,-1)
                predictions_.extend(predictions.cpu())
                labels_.extend(y.cpu())
                #--
                    
        #--- return eval
        pred_, recall_, f1_, acc_ = metric_object.eval_classification(labels_,predictions_)
        return pred_, recall_, f1_, acc_
            
            
    else:    
        model.eval()
        size = len(dataloader.dataset)
        losses = 0
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)

                #-- 
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                losses += (loss)
                #--
        return losses / len(dataloader)       
    



def test_for_ensemble_v1(dataloader, model, loss_fn,cfg, metric_object):
    '''
    v8
    Ensemble

    v7
    sentienlview added  
    
    
    v6
    in ddp(fabric) setting,     
    
    v4_3
    topview & steetview validation 
    if cfg.DATA_TYPE=="topview":
        pred = model(topview_img)
    elif cfg.DATA_TYPE =="streetview":
        pred = model(str_img)

    
    v4_2
    dataloader => 3 images 2 labels ()
    3 images : str_img, topview_img, _, 
    2 labels : y, original_labels
    
    
    v4_1
    dataloader => 3 images (street ,topview1, topview2)
    but use only topview1
    
    v4
    dataloader => 2 images (street and topview)
    
     
    '''
    if cfg.LOSS_FN == "CE" or "KL":
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, top1_acc, top5_acc = 0, 0, 0

        predictions_ = []
        labels_ = []
        print("#-------------- Start Validation")

        with torch.no_grad():
            for str_img, topview_img, sentinel_imgs, _, y in dataloader:
                
                if not cfg.FABRIC:
                    str_img, topview_img, sentinel_imgs,y = str_img.to(cfg.DEVICE), topview_img.to(cfg.DEVICE), sentinel_imgs.to(cfg.DEVICE),y.to(cfg.DEVICE)
                # # caution : model only load topview 
                # if cfg.DATA_TYPE=="topview":
                #     pred = model(topview_img)
                # elif cfg.DATA_TYPE =="streetview":
                #     pred = model(str_img)
                # elif cfg.DATA_TYPE =="sentinel2":
                #     pred = model(sentinel_imgs)
                    
                # elif cfg.DATA_TYPE =="ensemble":
                pred = model(str_img, topview_img, sentinel_imgs)

                #--
                predictions = torch.argmax(pred,-1)
                predictions_.extend(predictions.cpu())
                labels_.extend(y.cpu())
                #--
                    
        #--- return eval
        pred_, recall_, f1_, acc_ = metric_object.eval_classification(labels_,predictions_)
        return pred_, recall_, f1_, acc_
            
            
    else:    
        model.eval()
        size = len(dataloader.dataset)
        losses = 0
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)

                #-- 
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                losses += (loss)
                #--
        return losses / len(dataloader)       




def test_v9(dataloader, model, loss_fn,cfg, metric_object):
    '''
    v9
    alldata type addded
    
    v7
    sentienlview added  
    
    
    v6
    in ddp(fabric) setting,     
    
    v4_3
    topview & steetview validation 
    if cfg.DATA_TYPE=="topview":
        pred = model(topview_img)
    elif cfg.DATA_TYPE =="streetview":
        pred = model(str_img)

    
    v4_2
    dataloader => 3 images 2 labels ()
    3 images : str_img, topview_img, _, 
    2 labels : y, original_labels
    
    
    v4_1
    dataloader => 3 images (street ,topview1, topview2)
    but use only topview1
    
    v4
    dataloader => 2 images (street and topview)
    
     
    '''
    if cfg.LOSS_FN == "CE" or "KL":
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, top1_acc, top5_acc = 0, 0, 0

        predictions_ = []
        labels_ = []
        print("#-------------- Start Validation")

        with torch.no_grad():
            for str_img, topview_img, sentinel_imgs, _, y in dataloader:
                
                if not cfg.FABRIC:
                    str_img, topview_img, sentinel_imgs,y = str_img.to(cfg.DEVICE), topview_img.to(cfg.DEVICE), sentinel_imgs.to(cfg.DEVICE),y.to(cfg.DEVICE)
                # caution : model only load topview 
                if cfg.DATA_TYPE=="topview":
                    pred = model(topview_img)
                elif cfg.DATA_TYPE =="streetview":
                    pred = model(str_img)
                elif cfg.DATA_TYPE =="sentinel2":
                    pred = model(sentinel_imgs)
                    
                elif cfg.DATA_TYPE =="ensemble":
                    pred = model(str_img, topview_img)
                    
                elif cfg.DATA_TYPE =="alldata":
                    pred = model(str_img, topview_img, sentinel_imgs)
                #--
                predictions = torch.argmax(pred,-1)
                predictions_.extend(predictions.cpu())
                labels_.extend(y.cpu())
                #--
                    
        #--- return eval
        pred_, recall_, f1_, acc_ = metric_object.eval_classification(labels_,predictions_)
        return pred_, recall_, f1_, acc_
            
            
    else:    
        model.eval()
        size = len(dataloader.dataset)
        losses = 0
        with torch.no_grad():
            for X, y in dataloader:
                
                if not cfg.FABRIC:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(X)

                #-- 
                pred = pred.squeeze(-1)
                pred = pred.to(torch.float32)
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                losses += (loss)
                #--
        return losses / len(dataloader)       
    