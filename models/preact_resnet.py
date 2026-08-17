# File Name: preact_resnet.py
# Last Updated: July 29, 2026
# Description:
#   This file defines the neural network model used in our experiments on
#   CIFAR-10 (PreActResNet). It is a version of ResNet, a popular image
#   classification model. This file includes the model's building blocks,
#   the full model class, and a few ready-to-use sizes of the model
#   (PreActResNet18, 34, 50, 101, 152), as provided by Rice et al..
#   We use PreActResNet18 for our robust overfitting experiments.
# References:
#   * He, K., Zhang, X., Ren, S., and Sun, J. (2016). Identity Mappings in
#     Deep Residual Networks. ECCV.
#   * Rice, L., Wong, E., and Kolter, J. Z. (2020). Overfitting in
#     adversarially robust deep learning. ICML.

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """The basic building block used in the smaller models (18 and 34 layers).

    Each block normalizes and activates the data first, then runs it through
    two convolution layers. It also keeps a copy of the original input
    around to add back in at the end, which is what makes this a "residual"
    block.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # If the input and output don't match in size, we need an extra
        # conversion step so we can still add them together later.
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        # Use the converted version of the input if needed, otherwise just
        # use the original input as-is.
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut  # add the original input back in
        return out


class PreActBottleneck(nn.Module):
    """The building block used in the bigger models (50, 101, and 152 layers).

    Works the same way as the basic block above, but squeezes the data down
    before processing it and expands it back out after, which keeps things
    faster even as the model gets deeper.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        # Same idea as before: convert the input if its size doesn't match
        # the output, so we can add them together later.
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut  # add the original input back in
        return out


class PreActResNet(nn.Module):
    """The full model, built by stacking the blocks defined above.

    The model has 4 main stages. Each stage shrinks the image a bit and
    adds more channels, so the model can learn increasingly complex
    patterns as data moves through it. At the end, it pools everything
    down and makes a final prediction.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64  # keeps track of channel count as we add more layers

        # First layer that processes the raw image. CIFAR-10 images are
        # small, so we don't shrink the image here like some other versions
        # of this model do.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # The 4 main stages of the model. Each one (after the first) shrinks
        # the image size while increasing the number of channels.
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final normalization step before making the prediction.
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Builds one stage of the model out of several blocks in a row.

        Only the first block in the stage shrinks the image; the rest keep
        the image size the same.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # update for the next block
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        
        # Run the data through all 4 stages of the model.
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.relu(self.bn(out))
        
        # Shrink the data down to one value per channel, then make the
        # final prediction.
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# These functions below give us ready-to-use versions of the model at
# different sizes. Bigger numbers mean deeper (and slower) models.
# PreActResNet18 is what we use for our robust overfitting experiments.

def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)

def PreActResNet34(num_classes=10):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes)

def PreActResNet50(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes)

def PreActResNet101(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes=num_classes)

def PreActResNet152(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes=num_classes)