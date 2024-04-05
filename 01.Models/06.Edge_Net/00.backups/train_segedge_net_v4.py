import torch
import dsutils
import easydict
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import lightning as L
import albumentations as A
from transformers import AutoImageProcessor, AutoModel
from notebooks import telepix_utils
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

RESUME = 20000
#DEVICES = list(range(8))
MODEL_NAME = '/hdd1/ysyoon/models/seggpt_vit-large' if RESUME == -1 else f'ckpts/{RESUME:06d}'

BATCH_SIZE = 4
ROOT_DIR = '/hdd1/ysyoon/datasets/'
DATASET_NAMES = [
    'aihub-landcover-satellite-all',
    'aihub-satellite-object-cloud',
    'eorssd',
    'isaid',
    'open-earth-map',
    'orsi-sod'
]
AUGMENT = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(p=0.5),
    A.GaussNoise(p=0.5),
    A.Blur(p=0.5),
])
WEIGHT_DECAY = 0.01
LR = 1e-5
BETAS = [0.9, 0.999]
NUM_TRAINING_STEPS = 100000


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, dataset_names, augment, image_processor):
        self.root_dir = root_dir
        self.dataset_names = dataset_names
        self.augment = augment
        self.image_processor = image_processor

        self.num_datasets = len(self.dataset_names)
        self.image_paths = [sorted(glob(f'{root_dir}/{dname}/post/train/images/*')) for dname in dataset_names]
        self.num_images = [len(paths) for paths in self.image_paths]

    def __len__(self):
        return sum(self.num_images)

    def transform(self,x):
        custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),    
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])  
        ])
        y = custom_transform(x)
        
        return y

    def __getitem__(self, idx):
        ds_idx = np.random.randint(self.num_datasets)
        image_paths = self.image_paths[ds_idx]
        
        while True:
            prompt_image_path = image_paths[np.random.randint(self.num_images[ds_idx])]
            prompt_label_path = prompt_image_path.replace('/images/', '/labels/')
            
            prompt_image = np.array(Image.open(prompt_image_path))
            prompt_label = np.array(Image.open(prompt_label_path))

            unique_classes = np.unique(prompt_label)
            unique_classes = unique_classes[unique_classes>0]
            if len(unique_classes) > 0:
                break

        target_class = np.random.choice(unique_classes)
        while True:
            input_image_path = image_paths[np.random.randint(self.num_images[ds_idx])]
            input_label_path = input_image_path.replace('/images/', '/labels/')

            input_image = np.array(Image.open(input_image_path))
            input_label = np.array(Image.open(input_label_path))

            unique_classes = np.unique(prompt_label)
            if target_class in unique_classes:
                break
            
        
        prompt_aug = self.augment(image=prompt_image, mask=prompt_label)
        prompt_image, prompt_label = prompt_aug['image'], prompt_aug['mask']

        input_aug = self.augment(image=input_image, mask=input_label)
        input_image, input_label = input_aug['image'], input_aug['mask']

        
        prompt_label = np.where(prompt_label==target_class, 1, 0)
        input_label = np.where(input_label==target_class, 1, 0)
        
        palette = [[0,0,0], [255,255,255]]
        prompt_label = dsutils.segmentation.visualize_label(prompt_label, palette)
        input_label = dsutils.segmentation.visualize_label(input_label, palette)

        #------- edge
        edge = telepix_utils.label_to_edge(input_label, 3)
        edge = torch.tensor(edge)
        edge = edge.unsqueeze(0)
        edge = edge.double()

        return prompt_image, prompt_label, input_image, input_label, edge

    def collate_fn(self, batch):
        prompt_images, prompt_labels, input_images, input_labels, edges = zip(*batch)
        batch = self.image_processor(prompt_images, prompt_labels, input_images, input_labels, return_tensors='pt')
        batch['masks'] = self.image_processor.generate_mask(len(prompt_images))
        batch['edges'] = torch.stack([i for i in edges])
        return batch

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(p=0.5),
    A.GaussNoise(p=0.5),
    A.Blur(p=0.5),
])

root_dir = '/hdd1/ysyoon/datasets/'
dataset_names = [
    'aihub-landcover-satellite-all',
    'aihub-satellite-object-cloud',
    'eorssd',
    'isaid',
    'open-earth-map',
    'orsi-sod'
]

def get_param_groups(model, weight_decay):
    no_decay = ["bias", "bn", "ln", "norm"]
    param_groups = [
        {
            # apply weight decay
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            # not apply weight decay
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return param_groups




#--- SegGPT

seg_gpt = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)



#--- EdgeNet

class Edge_Net(nn.Module):
    def __init__(self):
        super(Edge_Net, self).__init__()
        #-- edge layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))       
        
        return x1,x2,x3

#------------------------
edge_net = Edge_Net()

# Path to the saved model weights
model_path = "./notebooks/tmp_ckpts/edge_net_epoch_layer_change_10.pt"  

# Load the saved model weights
edge_net.load_state_dict(torch.load(model_path))


class CombinedModel(nn.Module):
    def __init__(self, model1, model2):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
        # edge_net freeze
        for param in self.model2.parameters():
            param.requires_grad = False           

    # def save_pretrained(self,path):
    #     torch.save(model.state_dict(), PATH)

    def save_pretrained(self, path):
        # Save the model
        self.save_pretrained(path)
    

    
    def forward(self, batch ):

        #---------
        labels = batch['labels']
        labels = labels[:,:,448:,:] # slice only input labels not prompt labels
        labels = F.interpolate(labels,(512,512),mode='nearest')
        labels = labels.float()
        
        
        #----------
        edge_gt = batch['edges'] 
        edge_gt = edge_gt.float()
        
        batch.pop('edges',None)
        outputs = self.model1(**batch)
            
        pred = outputs.preds[:,:, 448:]
        # resize pred => 512
        pred = F.interpolate(pred,(512,512),mode='nearest')
        pred = pred.float()
        #print("pred.shape : ", pred.shape)
        
        #perceptual loss from edge_net
        layer_1_out,layer_2_out,layer_3_out = self.model2(pred)
        layer_1_gt ,layer_2_gt ,layer_3_gt  = self.model2(labels)
        loss_1 = torch.nn.functional.l1_loss(layer_1_out, layer_1_gt)
        loss_2 = torch.nn.functional.l1_loss(layer_2_out, layer_2_gt)
        loss_3 = torch.nn.functional.l1_loss(layer_3_out, layer_3_gt)

        #--- loss 
        loss_seg = outputs.loss
        loss_percept = loss_1 + loss_2 + loss_3
        
        return loss_seg,loss_percept



#----
import wandb
import logging
from tqdm import tqdm
from lightning.fabric import Fabric
import lightning as L 

DEVICES = [0,1,2,3]

fabric = L.Fabric(
    strategy='ddp',
    accelerator='cuda',
    devices=DEVICES,
)
fabric.launch()

# data
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
dataset = Dataset(ROOT_DIR, DATASET_NAMES, AUGMENT, image_processor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)
dataiter = iter(dataloader)

# model
seg_gpt = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = CombinedModel(seg_gpt, edge_net)
# model resume 
# Load model 
a_model_path = "ckpts/seg_edge_v6/segedge_034000" 
model.load_state_dict(torch.load(a_model_path))

param_groups = get_param_groups(model, WEIGHT_DECAY)
optimizer = torch.optim.AdamW(param_groups, lr=LR, betas=BETAS)

_, optimizer = fabric.setup(model, optimizer)

st = 34000
pbar = tqdm(range(st, NUM_TRAINING_STEPS+1), disable=not fabric.is_global_zero)
for st in pbar:
    fabric.barrier()
    try:
        batch = next(dataiter)
    except StopIteration:
        dataiter = iter(dataloader)
        batch = next(dataiter)

    optimizer.zero_grad()
    batch = batch.to(fabric.device)
    ## model(**batch)
    loss_seg,loss_percept  = model(batch)

    loss_percept_weight = 1/10
    loss = loss_seg + loss_percept * loss_percept_weight
    #loss = outputs.loss
    
    fabric.backward(loss)
    optimizer.step()

    if st % 10 == 0 and fabric.is_global_zero:
        log = {'st': st, 'loss': loss.item(), 'loss_seg' : loss_seg.item(), 'loss_percept' : loss_percept.item(), 'percept_loss_weight': loss_percept_weight}
        dsutils.log.log_csv(log, 'segpt_edges_log_v6.csv')
        pbar.set_postfix(log)

        if st % 2000 == 0:
            torch.save(model.state_dict(), f'./ckpts/seg_edge_v6/segedge_{st:06d}')
    











































