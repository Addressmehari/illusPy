import torch
import torch.nn as nn
import torch.nn.functional as F

class ShapeCNN(nn.Module):
    """
    3-Layer Convolutional Neural Network for shape classification and feature visualization.
    """
    def __init__(self):
        super(ShapeCNN, self).__init__()
        
        # Layer 1: Detects basic edge primitives
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128 -> 64x64
        
        # Layer 2: Combines edges into corner / curve patterns
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        
        # Layer 3: Higher-level shape detector
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        # Fully Connected Classifier
        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)  # Logit for shape class

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

    def get_feature_maps(self, x):
        """Returns intermediate feature maps from Conv layers."""
        f1 = F.relu(self.conv1(x))
        f1_pool = self.pool1(f1)
        
        f2 = F.relu(self.conv2(f1_pool))
        f2_pool = self.pool2(f2)
        
        f3 = F.relu(self.conv3(f2_pool))
        
        return {
            'conv1': f1,
            'conv2': f2,
            'conv3': f3
        }

# Alias for backwards compatibility
SquareCNN = ShapeCNN
