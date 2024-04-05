import numpy as np 
import matplotlib.pyplot as plt
#from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np 

def torch_display(img):
    
    img = img.permute(1,2,0)
    img = np.asarray(img)   
    
    fig_size= (10,10)
    plt.figure(figsize=fig_size)
    plt.imshow(img)
    
def torch_display_mask(mask):
    mask = mask.unsqueeze(0)
    
    mask = mask.permute(1,2,0)
    mask = np.asarray(mask)   
    
    fig_size= (10,10)
    plt.figure(figsize=fig_size)
    plt.imshow(mask)
    
    
    


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



def label_to_edge(seg_map,edge_width):

    seg_map = np.asarray(seg_map)
    #print("label to edge shape : ", seg_map.shape)
    
    # shape :  (512, 512, 3)
    #seg_map = seg_map[:,:,1].copy()
    #print("shape : ", seg_map.shape)    
    
    h,w = seg_map.shape
    edge = np.zeros((h, w), dtype=np.uint8)
    
    # attach label to zero 
    #ignore_index = 255
    ignore_index = 999
    
    # down
    edge_down = edge[1:h, :]
    edge_down[(seg_map[1:h, :] != seg_map[:h - 1, :])
              & (seg_map[1:h, :] != ignore_index) &
              (seg_map[:h - 1, :] != ignore_index)] = 1
    # left
    edge_left = edge[:, :w - 1]
    edge_left[(seg_map[:, :w - 1] != seg_map[:, 1:w])
              & (seg_map[:, :w - 1] != ignore_index) &
              (seg_map[:, 1:w] != ignore_index)] = 1
    # up_left
    edge_upleft = edge[:h - 1, :w - 1]
    edge_upleft[(seg_map[:h - 1, :w - 1] != seg_map[1:h, 1:w])
                & (seg_map[:h - 1, :w - 1] != ignore_index) &
                (seg_map[1:h, 1:w] != ignore_index)] = 1
    # up_right
    edge_upright = edge[:h - 1, 1:w]
    edge_upright[(seg_map[:h - 1, 1:w] != seg_map[1:h, :w - 1])
                 & (seg_map[:h - 1, 1:w] != ignore_index) &
                 (seg_map[1:h, :w - 1] != ignore_index)] = 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (edge_width, edge_width))
    
    edge = cv2.dilate(edge, kernel)

    #--------
    #stack_edge = np.stack([edge, edge, edge], axis=-1)

    #print("dd ", stack_edge.shape)
    return edge




def robust_weight_average(model_org, model, alpha):
    #-------------------------------------------
    
    theta_0 = model_org.state_dict()
    theta_1 = model.state_dict()
    
    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())
    
    
    # interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }
    
    # update the model acccording to the new weights
    model.load_state_dict(theta)

    # evaluate
    #evaluate(finetuned, args)

    return model 