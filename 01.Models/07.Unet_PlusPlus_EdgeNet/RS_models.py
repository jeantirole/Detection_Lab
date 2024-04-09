# Define your additional layers
import torch.nn as nn
import torch 
import segmentation_models_pytorch as smp 




class Edge_Net(nn.Module):
    def __init__(self,input_channel):
        super(Edge_Net, self).__init__()
        #-- load base
        # self.base_model = base_model

        self.input_channel = input_channel
        #-- edge layer
        self.conv1 = nn.Conv2d(in_channels=self.input_channel, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) 


    def forward(self, x):
        #outputs = self.base_model(input)
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.conv3(x2)        
        return x1,x2,x3



class CombinedModel(nn.Module):
    def __init__(self, model1, model2):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
        # seg_net train
        for param in self.model1.parameters():
            param.requires_grad = True       
        
        
        # edge_net freeze
        for param in self.model2.parameters():
            param.requires_grad = False           


    
    def forward(self, img, mask ):

        #---------       
        outputs = self.model1(img)
        
        #perceptual loss from edge_net
        layer_1_out,layer_2_out,layer_3_out = self.model2.forward_freeze(outputs)
        
        layer_1_gt ,layer_2_gt ,layer_3_gt  = self.model2.forward_freeze(mask)
        
        loss_1 = torch.nn.functional.l1_loss(layer_1_out, layer_1_gt,reduction='mean')
        loss_2 = torch.nn.functional.l1_loss(layer_2_out, layer_2_gt,reduction='mean')
        loss_3 = torch.nn.functional.l1_loss(layer_3_out, layer_3_gt,reduction='mean')
        loss_3 = loss_3 * 0.1
        
        #--- loss 
        
        loss_percept = loss_1 + loss_2 + loss_3
        
        return outputs,loss_percept
    
    
    def forward_eval(self,img):
        
        outputs = self.model1(img)
        
        return outputs