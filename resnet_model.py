import multiprocessing
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
torch.manual_seed(73647)

# Class definiition for a BasicBlock with a skipped connection
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel, skip_kernel, stride=1, bias=True, skip_conn = False):
        super(BasicBlock, self).__init__()

        # Layer1: conv -> BN -> ReLU
        self.conv1 = nn.Conv2d(in_channels = in_planes,
                               out_channels = planes,
                               kernel_size=kernel[0],
                               stride=stride,
                               padding=kernel[1],
                               bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

        # Layer2: conv -> BN -> ReLU
        self.conv2 = nn.Conv2d(in_channels = planes,
                               out_channels = planes,
                               kernel_size=kernel[0],
                               stride=1,
                               padding=kernel[1],
                               bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

        # Including a skipped connection which will help the layers of varying dimensions connect to each other
        if skip_conn:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = in_planes,
                          out_channels = planes,
                          kernel_size=skip_kernel[0],
                          padding=skip_kernel[1],
                          stride=stride,
                          bias=bias),
                nn.BatchNorm2d(planes))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Include a skipped connection
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

# Class Definition for the ResNet defined for CIFAR-10 class
class ResNet(nn.Module):
    def __init__(self, block, channels, num_layers, kernel, skip_kernel, num_classes=10, bias=True):
        super(ResNet, self).__init__()
        self.kernel = kernel
        self.skip_kernel = skip_kernel
        self.channels = channels
        self.num_layers = num_layers

        # Initialize the 3 channels into conv->BN->ReLU
        self.layer0 = self.make_initial_layer(3,self.channels,kernel[0],kernel[1], bias)

        # Layer with 2 ResNet blocks, maps 32 to 32 channels
        self.layer1 = self.make_layer(block, self.channels, 2, stride=1, bias=bias)

        # Layer with 4 ResNet blocks, maps 32 to 64 channels
        self.layer2 = self.make_layer(block, 2*self.channels, 4, stride=2, bias=bias)

        # Layer with 8 ResNet blocks, maps 64 to 128 channels
        self.layer3 = self.make_layer(block, 2*self.channels, 8, stride=2, bias=bias)

        # Layer with 2 ResNet blocks, maps 128 to 256 channels
        self.layer4 = self.make_layer(block, 2*self.channels, 2, stride=2, bias=bias)

        # Layer to map 256 to 10 output channels
        self.linear_class_mapper = nn.Linear(self.layer4[-1].bn2.num_features, num_classes)

        # Layer to aggregate data
        self.avg_pool2d = nn.AvgPool2d(4)

        # Path to model file
        self.modelpath = "./mini_project_model.pt"

    def make_initial_layer(self, in_channels, out_channels, kernel_size, padding , bias):
        return nn.Sequential( nn.Conv2d(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding= padding,
                                        bias=bias),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU())

    def make_layer(self, block, planes, num_blocks, stride, bias=True):
        # Create an initial block with lesser input to higher output channels
        layer_blocks = [block(self.channels, planes,self.kernel,self.skip_kernel, stride, bias, True)]
        self.channels = planes
        # Iteratively create Subsequent blocks with equal input and output channels
        for block_iter in range(num_blocks-1):
            layer_blocks.append(block(self.channels, planes,self.kernel,self.skip_kernel, 1, bias, False))
        return nn.Sequential(*layer_blocks)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_class_mapper(out)
        return out

    def saveModel(self):
        torch.save(self.state_dict(), self.modelpath)

    def loadModel(self):
        self.load_state_dict(torch.load(self.modelpath))

def mini_project_model():
    return ResNet(BasicBlock, 32, 4,kernel=(3,1),skip_kernel=(1,0), num_classes=10, bias=True)
