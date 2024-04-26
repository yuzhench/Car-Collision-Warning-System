"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
import torch.nn as nn
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss_transfer import YoloLoss

from transfer_learning import get_bboxes_transfer, mean_average_precision_transfer

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
TRANSFER_MODEL = "transfer.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    print("size of the loop: ", len(loop))
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        new_shape = (16,7,7,40)#modify
        new_lables = torch.zeros(new_shape)#modify
        new_lables[...,:20] = y[...,:20]
        new_lables[...,30:] = y[...,20:]
        y = new_lables

        
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        print("the shape of y is:", y.shape)
         

        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")



def transfer_learning_train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # Forward pass
        out = model(x)
        
        # Calculate loss
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")




def main():
    # model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    print("the device we use is: ", DEVICE)
    model = Yolov1().to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        # "data/bigger_dataset.csv",
        "data/100examples.csv",
        transform=transform,
        img_text_dir=(IMG_DIR,LABEL_DIR)
    )

    # test_dataset = VOCDataset(
    #     "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR, ##-->current command out 
    # )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,##-->current command out 
    #     pin_memory=PIN_MEMORY,
    #     shuffle=True,
    #     drop_last=True,
    # )

    #load the model ----------------------------
    # model = Yolov1()
    model_weights_path = "overfit_2000.pth.tar"
    checkpoint = torch.load(model_weights_path)
    model.load_state_dict(checkpoint['state_dict'])

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc_layer1.parameters():
        param.requires_grad = True
    for param in model.fc_layer2.parameters():
        param.requires_grad = True

    # model.fc_layer1 = nn.Linear(in_features=50176, out_features=496).to(DEVICE)
    model.fc_layer2 = nn.Linear(in_features=496, out_features= 1960).to(DEVICE)
    #load_the_model------------------------------

    for epoch in range(EPOCHS):
        # for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(8):
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        #    import sys
        #    sys.exit()

        pred_boxes, target_boxes = get_bboxes_transfer(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        # print("pred_boxes is: ", pred_boxes)
        print("pred_boxes is: ", pred_boxes)


        mean_avg_prec = mean_average_precision_transfer(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        #save the model in the location
        if mean_avg_prec > 0.9:
           checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
           }
           save_checkpoint(checkpoint, filename=TRANSFER_MODEL)
           import time
           time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn)
        # transfer_learning_train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
