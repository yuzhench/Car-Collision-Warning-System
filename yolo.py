import torch
import tensor
import torch.nn as nn

class YOLO_CNN(nn.Module):
    def __init__(self):
        super(YOLO_CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192,kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128,kernel_size=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=1,stride=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,stride=1)


        #below 2 layers repeat 4 times:
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.conv8 = nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3, stride=1)
        #######

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1,stride=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1)

        #below 2 layers repeat 2 times:
        self.conv11 = nn.Conv2d(in_channels=1024,out_channels=512, kernel_size=1,stride=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1)
        #######

        self.conv13 = nn.Conv2d(in_channels=1024,out_channels=1024, kernel_size=3, stride=1)
        self.conv14 = nn.Conv2d(in_channels=1024,out_channels=1024, kernel_size=3, stride=2)
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1)
        self.conv16 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3,stride=1)
        
        #the pool layer 
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)

        #the fully connection layer 
        self.fc_layer1 = nn.Linear(50176,4096)
        self.fc_layer2 = nn.Linear(4096,1470)


    def forward(self,x):
        x = self.max_pool(torch.relu(self.conv1(x))) #conv1 and max pool 
        x = self.max_pool(torch.relu(self.conv2(x))) #conv2 and max pool

        x = torch.relu(self.conv3(x)) # conv3
        x = torch.relu(self.conv4(x)) # conv4
        x = torch.relu(self.conv5(x)) # conv5
        x = torch.relu(self.conv6(x)) # conv6

        #repeat 4 times 
        x = torch.relu(self.conv7(x)) #conv7
        x = torch.relu(self.conv8(x)) #conv8
        x = torch.relu(self.conv7(x)) #conv7
        x = torch.relu(self.conv8(x)) #conv8
        x = torch.relu(self.conv7(x)) #conv7
        x = torch.relu(self.conv8(x)) #conv8
        x = torch.relu(self.conv7(x)) #conv7
        x = torch.relu(self.conv8(x)) #conv8

        x = torch.relu(self.conv9(x)) #conv9
        x = self.max_pool(torch.relu(self.conv10(x))) #conv10 and max pool

        #repeat 2 times
        x = torch.relu(self.conv11(x)) #conv11
        x = torch.relu(self.conv12(x)) #conv12
        x = torch.relu(self.conv11(x)) #conv11
        x = torch.relu(self.conv12(x)) #conv12

        x = torch.relu(self.conv13(x)) #conv13
        x = torch.relu(self.conv14(x)) #conv14
        x = torch.relu(self.conv15(x)) #conv15
        x = torch.relu(self.conv16(x)) #conv16

        #flatten the layer 
        x = tensor.view(-1)

        #first fully connection layer 
        x = self.fc_layer1(50176,4096)
        x = self.fc_layer2(4096, 1470)

        

