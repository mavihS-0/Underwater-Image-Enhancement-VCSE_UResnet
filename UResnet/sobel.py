import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import time

class Edge_Detector(nn.Module):
    def __init__(self):
        super(Edge_Detector, self).__init__()
        # Using a Sobel filter for edge detection instead of a fixed kernel
        self.sobel_x = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_kernel_x = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]])
        sobel_kernel_y = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]])
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x.repeat(3, 1, 1, 1), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y.repeat(3, 1, 1, 1), requires_grad=False)
    
    def forward(self, x):
        x_gray = torch.mean(x, dim=1, keepdim=True)  # Convert RGB to grayscale
        edge_x = self.sobel_x(x_gray)
        edge_y = self.sobel_y(x_gray)
        edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge_map


class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block,self).__init__()
        self.conv=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu=nn.ReLU(inplace=True)
        self.bn=nn.BatchNorm2d(64)
    def forward(self,x):
        
        return torch.add(self.bn(self.conv(self.relu(self.bn(self.conv(x))))),x)

class UResNet_Sob(nn.Module):
    def __init__(self):
        super(UResNet_Sob,self).__init__()
        self.edge_detector=Edge_Detector()
        self.residual_layer=self.stack_layer(Res_Block,16)
        self.input=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.output=nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu=nn.ReLU(inplace=True)
#         for m in self.modules():
#             if isinstance(m,nn.Conv2d):
#                 n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
#                 m.weight.data.norm(0,sqrt(2./n))
    def stack_layer(self,block,num_of_layers):
        layers=[]
        for _ in range(num_of_layers):
            layers.append(block())
        return nn.Sequential(*layers)
    def forward(self,x):        
        x=self.relu(self.input(x))
        out=self.residual_layer(x)
        out=torch.add(out,x)
        out=self.output(out)
        edge_map=self.edge_detector(out)
        #out=torch.add(out,residual)
        return out,edge_map