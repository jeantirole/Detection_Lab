import matplotlib.pyplot as plt 
import numpy as np 
import torch

def nms_pytorch(P : torch.tensor ,thresh_iou : float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """
 
    # we extract coordinates for every 
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]
 
    # we extract the confidence scores as well
    scores = P[:, 4]
 
    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
     
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()
 
    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
     
 
    while len(order) > 0:
         
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]
 
        # push S in filtered predictions list
        keep.append(P[idx])
 
        # remove S from P
        order = order[:-1]
 
        # sanity check
        if len(order) == 0:
            break
         
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)
 
        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
 
        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
         
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
 
        # find the intersection area
        inter = w*h
 
        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim = 0, index = order) 
        
        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
         
        # find the IoU of every prediction in P with S
        IoU = inter / union
 
        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]
     
    return keep

# Visualize
import matplotlib.patches as patches

def vis(data):

    P = data
    # Create a figure and axis
    fig, ax = plt.subplots()


    color_pallete= ['red','black','green','cyan']

    # Plot bounding boxes
    for i,box in enumerate(P):
        x, y, w, h, confidence = box.tolist()
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='red', facecolor='none', label=f'Conf: {confidence:.2f}')
        ax.add_patch(rect)

    # Set axis limits based on the bounding boxes
    ax.set_xlim(0,max([max(i) for i in P]))
    ax.set_ylim(0,max([max(i) for i in P]))

    # Display legend
    ax.legend()

    # Set labels
    plt.title('Bounding Boxes Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show the plot
    plt.show()



# Open the file in read mode
with open('/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Data/NMS_Test_Data/labels.txt', 'r') as file:
    # Read lines from the file
    lines = file.readlines()

# Initialize an empty list to store the parsed data
data = []

# Iterate over each line and parse the values
for line in lines:
    values = list(map(float, line.split()))
    data.append(values)

# Convert the list of lists to a NumPy array for further processing if needed
import numpy as np
data_array = np.array(data)

# Display the parsed data
print(data_array)



#--- image 
from PIL import Image

img_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Data/NMS_Test_Data/sample.jpg"
img = Image.open(img_path)
img = img.resize((620,620))

#--- 
boxes = data_array

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the image
ax.imshow(img)

# Plot bounding boxes
for box in boxes:
    x, y, w, h, _ = box
    rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()

#---
boxes = torch.tensor(boxes)
filtered_boxes = nms_pytorch(boxes,0.5)

#---
vis(filtered_boxes)