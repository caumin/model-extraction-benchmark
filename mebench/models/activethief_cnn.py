"""ActiveThief CNN architecture implementation."""

import torch
import torch.nn as nn
from typing import List

class ActiveThiefCNN(nn.Module):
    """
    CNN architecture used in "ActiveThief: Model Extraction Using Active Learning" (AAAI 2020).
    
    Structure:
    - l=3 blocks.
    - Each block: [Conv3x3-ReLU-BN] x 2 -> MaxPool2x2 -> Dropout.
    - Filters: 32 -> 64 -> 128.
    - Classifier: Flatten -> Linear(num_classes).
    
    Ref: "Consists of l convolution blocks (default l=3). Each block: 2 repeated units of 
          [2 convolution layers (3x3 kernel, stride 1) + ReLU + BatchNorm] 
          followed by 1 pooling layer (2x2 kernel, stride 2) and dropout."
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout_prob = dropout_prob
        
        # Block 1: 32 filters
        self.block1 = self._make_block(input_channels, 32, dropout_prob)
        # Block 2: 64 filters
        self.block2 = self._make_block(32, 64, dropout_prob)
        # Block 3: 128 filters
        self.block3 = self._make_block(64, 128, dropout_prob)
        
        # Classifier
        # Calculate input size for linear layer dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 32, 32) # Assume 32x32 base, or we could add input_size arg
            # But since standard CIFAR/SVHN are 32x32, and MNIST is 28x28 (padded usually or not), 
            # we should probably handle the specific dataset size.
            # However, the factory doesn't pass input_size.
            # Let's use LazyLinear if possible, OR use a fixed size for CIFAR (32x32).
            # Given we are reproducing for CIFAR-10 primarily here.
            pass

        # Use LazyLinear to support multiple input resolutions (32x32, 28x28) automatically.
        # To ensure optimizer works, we run a dummy forward if possible, 
        # but factory doesn't know the size.
        # Fallback: Use 32x32 calculation as default (2048) but allow mismatch? No.
        # Let's use a standard Linear with 2048 input features (ActiveThief paper uses 32x32).
        # If input is different, it will crash. This is better than silent failure.
        
        self.flatten = nn.Flatten()
        # 128 * 4 * 4 = 2048 for 32x32 input
        self.fc = nn.Linear(2048, num_classes)

    def _make_block(self, in_channels: int, out_channels: int, p: float) -> nn.Sequential:
        return nn.Sequential(
            # Unit 1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            
            # Unit 2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            
            # Pool + Dropout
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=p) if p > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        
        out = self.flatten(out)
        out = self.fc(out)
        return out

