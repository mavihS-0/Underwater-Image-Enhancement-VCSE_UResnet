import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import time
import torch.nn.functional as F


class Edge_Detector(nn.Module):
    def __init__(self):
        super(Edge_Detector, self).__init__()
        # Define a single convolution layer for edge detection
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        
        # Initialize weights for edge detection filter
        nn.init.constant_(self.conv1.weight, 1)  # Set all weights to 1
        nn.init.constant_(self.conv1.weight[0, 0, 1, 1], -8)  # Set specific weights to -8 for edge effect
        nn.init.constant_(self.conv1.weight[0, 1, 1, 1], -8)
        nn.init.constant_(self.conv1.weight[0, 2, 1, 1], -8)

    def forward(self, x1):
        # Apply the convolution to obtain the edge map
        edge_map = self.conv1(x1)
        return edge_map


class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        # Define a convolutional layer, ReLU activation, and BatchNorm
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(64)
    
    def forward(self, x):
        # Apply Conv -> ReLU -> BatchNorm twice, then add the input (residual connection)
        return torch.add(self.bn(self.conv(self.relu(self.bn(self.conv(x))))), x)


class UResNet_P(nn.Module):
    def __init__(self):
        super(UResNet_P, self).__init__()
        # Initialize edge detector and residual layers
        self.edge_detector = Edge_Detector()
        
        # Create 16 residual blocks in sequence
        self.residual_layer = self.stack_layer(Res_Block, 16)
        
        # Define input and output convolutional layers
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
    def stack_layer(self, block, num_of_layers):
        # Helper function to stack multiple layers of residual blocks
        layers = []
        for _ in range(num_of_layers):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Pass input through the initial convolution and activation
        x = self.relu(self.input(x))
        
        # Process through stacked residual layers
        out = self.residual_layer(x)
        
        # Add the input to the residual output (residual connection)
        out = torch.add(out, x)
        
        # Pass through the output convolutional layer
        out = self.output(out)
        
        # Generate edge map from the output using the edge detector
        edge_map = self.edge_detector(out)
        
        return out, edge_map