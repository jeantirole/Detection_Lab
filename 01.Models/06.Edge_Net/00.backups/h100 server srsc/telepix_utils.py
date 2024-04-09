import cv2
import numpy as np 

def label_to_edge(seg_map,edge_width):

    seg_map = np.asarray(seg_map)
    #print("label to edge shape : ", seg_map.shape)
    
    # shape :  (512, 512, 3)
    seg_map = seg_map[:,:,1].copy()
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