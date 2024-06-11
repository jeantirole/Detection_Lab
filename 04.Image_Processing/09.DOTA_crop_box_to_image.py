import pandas as pd 
import numpy as np 
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
import cv2
from tqdm import tqdm 

# load img n ann 
img_folder = "/mnt/hdd/eric/.tmp_ipy/00.Data/DOTA_dataset/dota1_origin/val/images"
ann_folder = "/mnt/hdd/eric/.tmp_ipy/00.Data/DOTA_dataset/dota1_origin/val/annfiles"
img_files = sorted(glob(img_folder+"/*.png"))
ann_files = sorted(glob(ann_folder+"/*.txt"))


cnt = 0
#--- load each image 
for index_ in tqdm( range(len(img_files)), desc="satellite image chip maker"):
    print("img index : ", index_)
    img = img_files[index_]
    ann = ann_files[index_]
    
    # open image 
    image = cv2.imread(img)

    # open ann 
    with open(ann) as f:
        ann_lines = f.readlines()
    print(len(ann_lines))


    #try:
    for index_line_ in range(len(ann_lines)):
        #index_line_ = 0
        single_ann_line = ann_lines[index_line_]
        poly_box = [float(i) for i in ann_lines[index_line_].split()[0:8]]
        label_class = ann_lines[index_line_].split()[8]
        #print(poly_box)
        #print(label_class)

        # polys 
        vertices = np.asarray(poly_box)
        for vi in range(len(vertices)):
            if vertices[vi] <0:
                vertices[vi] = 0
        vertices = vertices.reshape((4,2))
        polygon_coords = vertices

        # Create a Shapely Polygon object
        polygon = Polygon(polygon_coords)

        # Create a mask image with the same shape as the original image
        mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

        # Draw the filled polygon on the mask
        cv2.fillPoly(mask, [np.array(polygon_coords, dtype=np.int32)], color=255)

        # Bitwise AND operation to get the cropped region
        cropped_image = cv2.bitwise_and(image, image, mask=mask)

        # mask to rectangle
        xs, ys = [i[0] for i in vertices], [i[1] for i in vertices]
        x_max,x_min = int(max(xs)), int(min(xs))
        y_max,y_min = int(max(ys)), int(min(ys))

        cropped_image_rec = cropped_image[ y_min:y_max, x_min:x_max, :]

        # Save the result
        dst_folder = "/mnt/hdd/eric/.tmp_ipy/00.Data/DOTA_dataset_chips/v2"
        sep = "images"
        file_path = f"crops_img_{index_}_instance_{index_line_}_.png"
        img_result = os.path.join(dst_folder, sep, file_path)    
        cv2.imwrite(img_result,cropped_image_rec)
        
        # Specify the file path
        dst_folder = "/mnt/hdd/eric/.tmp_ipy/00.Data/DOTA_dataset_chips/v2"
        sep = "anns"
        file_path = f"crops_ann_{index_}_instance_{index_line_}_.txt"
        ann_result = os.path.join(dst_folder, sep, file_path)
        with open(ann_result, 'w') as file:
            file.write(single_ann_line)
    # except:
    #     pass

        #plt.imshow(cropped_image_rec)
        #cv2.imshow('Cropped Image', cropped_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
