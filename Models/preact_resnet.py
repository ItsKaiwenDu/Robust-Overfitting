'''Pre-activation ResNet in PyTorch.

Pre-activation is a design modification for Residual Networks (ResNets). Normally, a ResNet 
applies layers in order of: Convolutions -> Batch Normalization -> Activation (ReLU). 
In a Pre-activation ResNet, we rearrange this sequence so that Batch Normalization and 
Activation are applied *before* Convolutional weight layers.

By putting normalization and activation first, network maintains a clean "shortcut highway"
that lets raw information pass directly from early layers to later layers. This:
1. Improves gradient flow during learning (meaning feedback signals used to update 
   model's weights can travel backward through network's layers without getting lost).
2. Prevents vanishing or exploding learning signals (where feedback values either shrink 
   to zero, stopping all learning, or grow too large, causing training to crash).
3. Increases overall training stability, making it much easier to train model to 
   defend against "noisy" images designed to trick it (adversarial training).

References:
1. Original Architecture Design: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
   "Identity Mappings in Deep Residual Networks." arXiv:1603.05027 (Introduced this pre-activation ResNet structure).
2. Project Context & Baseline: Leslie Rice, Eric Wong, and J. Zico Kolter.
   "Overfitting in Adversarially Robust Deep Learning." ICML 2020 (Adopted this PreActResNet-18 architecture for robust overfitting benchmarks, which this project replicates).
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of BasicBlock.
    In a pre-activation block, batch normalization and ReLU activation
    are applied *before* convolutional layers, which has been shown
    to improve gradient flow and reduce training instability.
    '''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        # Pre-activation normalization and activation layers
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # If spatial dimensions or channel counts change, project shortcut connection
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        # Apply pre-activation: BN -> ReLU
        out = F.relu(self.bn1(x))
        # Use pre-activated feature representation as shortcut projection source if needed
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # First convolution
        out = self.conv1(out)
        # Second convolution after BN and ReLU activation
        out = self.conv2(F.relu(self.bn2(out)))
        # Add residual connection
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of original Bottleneck module.
    Similar to BasicBlock but uses a 1x1, 3x3, 1x1 conv structure
    to reduce and then restore channel dimensions. Typically used for
    deeper networks (e.g., ResNet-50, 101, 152).
    '''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        # If spatial dimensions or channel counts change, project shortcut connection
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        # Apply pre-activation: BN -> ReLU
        out = F.relu(self.bn1(x))
        # Use pre-activated features for shortcut projection
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # 1x1 channel reduction conv
        out = self.conv1(out)
        # 3x3 spatial conv
        out = self.conv2(F.relu(self.bn2(out)))
        # 1x1 channel expansion conv
        out = self.conv3(F.relu(self.bn3(out)))
        # Add residual connection
        out += shortcut
        return out


class PreActResNet(nn.Module):
    '''General Pre-activation ResNet module.
    Takes a block type and a list of block counts per layer/stage to construct architecture.
    '''
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        # Initial convolutional layer mapping inputs (e.g., 3-channel CIFAR-10) to 64 feature maps
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Build four standard ResNet layer stages
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final normalization block before pooling and classifier
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        '''Helper function to build a sequential block stage with specified block type and counts.'''
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolutional layer
        out = self.conv1(x)
        
        # Execute basic residual block stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Final pre-activation normalization and activation
        out = F.relu(self.bn(out))
        
        # Global average pooling and classification head
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(num_classes=10):
    '''Constructs a Pre-activation ResNet-18 model (commonly used baseline for robust overfitting).'''
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)

def PreActResNet34(num_classes=10):
    '''Constructs a Pre-activation ResNet-34 model.'''
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes)

def PreActResNet50(num_classes=10):
    '''Constructs a Pre-activation ResNet-50 model.'''
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes)

def PreActResNet101(num_classes=10):
    '''Constructs a Pre-activation ResNet-101 model.'''
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes=num_classes)

def PreActResNet152(num_classes=10):
    '''Constructs a Pre-activation ResNet-152 model.'''
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes=num_classes)
