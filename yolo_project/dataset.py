import numpy as np
import torch
import pandas as pd
import os
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_text_dir, S=7, B=2,C=20,transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.img_text_dir = img_text_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self,index):
        #the direction of the text file in the local direction 
        text_path = os.path.join(self.img_text_dir[1], self.csv_file.iloc[index, 1])
        #create a container to store the box in each text file (since ther may be more than one box in one text file)
        boxes = []
        with open(text_path) as f:
            for line in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in line.replace("\n", "").split()
                ]

                boxes.append([class_label,x,y,width,height])

        #start to load the image from the local direction 
        image_path = os.path.join(self.img_text_dir[0],self.csv_file.iloc[index,0])
        image = Image.open(image_path) # this image will be the output 
        boxes = torch.tensor(boxes)

        if self.transform is not None:
            image, boxes = self.transform(image, boxes)

        #use the data in the boxes to generate the final label_matrix 
        #the shape of the label_matrix is (7,7,30) or (S,S, B*5+C)
        label_matrix = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix



