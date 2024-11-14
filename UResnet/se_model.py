import torch
import torch.nn as nn
import torch.nn.functional as F

class Edge_Detector(nn.Module):
    def __init__(self):
        super(Edge_Detector, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        nn.init.constant_(self.conv1.weight, 1)
        nn.init.constant_(self.conv1.weight[0, 0, 1, 1], -8)
        nn.init.constant_(self.conv1.weight[0, 1, 1, 1], -8)
        nn.init.constant_(self.conv1.weight[0, 2, 1, 1], -8)
      
    def forward(self, x):
        edge_map = self.conv1(x)
        return edge_map
class SE_Block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE_Block, self).__init__()
        # Fully connected layers for the squeeze-and-excitation operation
        # First layer reduces channel dimensions by the 'reduction' factor
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        
        # Second layer restores the channel dimensions back to the original
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        
        # Global average pooling layer to squeeze spatial dimensions to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        # Squeeze operation: apply global average pooling to reduce spatial dimensions
        se = self.avg_pool(x)
        
        # First fully connected layer with ReLU activation
        se = F.relu(self.fc1(se))
        
        # Second fully connected layer with Sigmoid activation to create the attention weights
        se = torch.sigmoid(self.fc2(se))
        
        # Scale the input feature map by the learned attention weights
        return x * se


class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.se_block = SE_Block(64)  # SE block added within the residual block

    def forward(self, x):
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se_block(x)  # Apply SE block on the residual output
        return torch.add(x, res)

class UResNet_SE(nn.Module):
    def __init__(self):
        super(UResNet_SE, self).__init__()
        self.edge_detector = Edge_Detector()  # Ensure edge_detector is initialized
        self.residual_layer = self.stack_layer(Res_Block, 16)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def stack_layer(self, block, num_of_layers):
        layers = [block() for _ in range(num_of_layers)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.input(x))
        out = self.residual_layer(x)
        out = torch.add(out, x)
        out = self.output(out)
        edge_map = self.edge_detector(out)  # Access edge_detector here
        return out, edge_map
