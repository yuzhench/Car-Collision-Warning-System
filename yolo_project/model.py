import torch
import torch.nn as nn


#define a unit of a single layer combine: conv2d & 
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    

class Yolov1(nn.Module):
    def __init__(self):
        super(Yolov1, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.darknet = self._create_conv_layers()

        # self.fc_layer1 = nn.Linear(50176, 4096)
        # self.fc_layer2 = nn.Linear(4096, 1470)
        self.fc_layer1 = nn.Linear(50176, 496)
        self.fc_layer2 = nn.Linear(496, 1470)
        self.drop = nn.Dropout(0.0)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.darknet(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = self.fc_layer1(x)
        x = self.drop(x)
        x = self.leakyrelu(x)
        x = self.fc_layer2(x)
        return x

    def _create_conv_layers(self):
        layers = nn.Sequential(
            CNNBlock(3, 64, kernel_size=7, stride=2),
            self.max_pool,
            CNNBlock(64, 192, kernel_size=3, stride=1),
            self.max_pool,
            CNNBlock(192, 128, kernel_size=1, stride=1),
            CNNBlock(128, 256, kernel_size=3, stride=1),
            CNNBlock(256, 256, kernel_size=1, stride=1),
            CNNBlock(256, 512, kernel_size=3, stride=1),
            self.max_pool,
            CNNBlock(512, 256, kernel_size=1, stride=1),
            CNNBlock(256, 512, kernel_size=3, stride=1),
            CNNBlock(512, 256, kernel_size=1, stride=1),
            CNNBlock(256, 512, kernel_size=3, stride=1),
            CNNBlock(512, 256, kernel_size=1, stride=1),
            CNNBlock(256, 512, kernel_size=3, stride=1),
            CNNBlock(512, 256, kernel_size=1, stride=1),
            CNNBlock(256, 512, kernel_size=3, stride=1),
            CNNBlock(512, 512, kernel_size=1, stride=1),
            CNNBlock(512, 1024, kernel_size=3, stride=1),
            self.max_pool,
            CNNBlock(1024, 512, kernel_size=1, stride=1),
            CNNBlock(512, 1024, kernel_size=3, stride=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1),
            CNNBlock(512, 1024, kernel_size=3, stride=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=2),
            CNNBlock(1024, 1024, kernel_size=3, stride=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=1)
        )
        return layers
