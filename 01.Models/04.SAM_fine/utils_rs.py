import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical



def torch_display(img):
    
    img = img.permute(1,2,0)
    img = np.asarray(img)   
    
    fig_size= (10,10)
    plt.figure(figsize=fig_size)
    plt.imshow(img)
    
    

def mask_display(label=None, nrows=None, ncols=None, channel_order=None):


    fig_size = (16,16)
    fig,axs = plt.subplots(nrows=nrows, ncols=nrows, figsize=fig_size)

    cnt = 0
    for row in range(nrows):
        for col in range(nrows):
            
            if channel_order =="torch":
                axs[row,col].imshow(label[cnt,:,:] )
                cnt += 1
            elif channel_order =="None":
                axs[row,col].imshow(label[:,:,cnt] )
                cnt += 1


    # Adjust layout to prevent overlap of subplots
    plt.tight_layout()

    # Show the plots
    plt.show()
    




def __mask_encoding__(self, label):
    
    label = np.asanyarray(label)
    
    zero_label = np.zeros((label.shape[0],label.shape[1],label.shape[2]))
    
    for k,v in self.palette.items():
        indices = np.where(np.all(label == v, axis=-1))
        zero_label[indices] = k 
    
    zero_label_ = zero_label[:,:,0].copy()

    label_oh = to_categorical(zero_label_,num_classes= len(self.palette.keys()) )     
    
    return label_oh
