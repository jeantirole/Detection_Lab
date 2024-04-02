import torch
import swin
import upper_net_mmseg
import torch.nn as nn 


#----------------------------------------------
class SamRS(torch.nn.Module):
    def __init__(self, 
                model1=None,
                model2=None, 
                decoder_use_batchnorm: bool = True,
                decoder_channels  = (512, 256, 128, 64), #(256, 128, 64, 32, 16),
                decoder_attention_type = None,
                classes1: int = 18,
                classes2: int = 20,
                classes3: int = 37,
                activation: str = None,
                aux_params: dict = None):
        
        super(SamRS,self).__init__()
        
        self.encoder = model1 
        self.decoder = model2
        self.semseghead_1 = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(self.encoder.out_channels[2], classes1, kernel_size=1)
            )

        # self.semseghead_2 = nn.Sequential(
        #         nn.Dropout2d(0.1),
        #         nn.Conv2d(self.encoder.out_channels[2], classes2, kernel_size=1)
        #     )

        # self.semseghead_3 = nn.Sequential(
        #         nn.Dropout2d(0.1),
        #         nn.Conv2d(self.encoder.out_channels[2], classes3, kernel_size=1)
        #     )
    
    def forward(self,x):
        features = self.encoder(x)
        output = self.decoder(*features)
        output_1 = self.semseghead_1(output)
        #output_2 = self.semseghead_2(output)
        #output_3 = self.semseghead_3(output)
        
        
        return output_1

#---------------------------------------    
            