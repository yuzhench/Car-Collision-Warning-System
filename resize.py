import numpy as py 
from PIL import Image
import os
import re
import cv2

image_direction = "/home/yuzhen/Desktop/yiting_yizhou_group/bilstm/bilstm_frames/"
destination_dir = "/home/yuzhen/Desktop/EECS373/EECS373_final_project/video_frames"
files = os.listdir(image_direction)

page_number = {250,253,256,259,262,265,268,271,274,277}

for file in files:
    file = os.path.join(image_direction, file)
    print(file)
    match = re.search(r"\d+", file)
    number = int(match.group())
    if number in page_number:
        img = cv2.imread(file)
        img = cv2.resize(img,(int(img.shape[1]/8), int(img.shape[0]/8)))
        file_name = os.path.join(destination_dir,str(number)+".jpg")
        cv2.imwrite(file_name,img)
        

 


