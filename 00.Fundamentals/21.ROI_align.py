img_path = "/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Data/sample_images/000001.jpg"
# bilinear interpolation to image 
import cv2
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(500,500))
print("ddd")