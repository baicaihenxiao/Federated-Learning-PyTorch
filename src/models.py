#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


NUM_CLASSES = 10


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class ResNet18Cifar(nn.Module):
    def __init__(self, args):
        super(ResNet18Cifar, self).__init__()
        self.model = resnet18(weights=None,
                              norm_layer=_resnet_norm_layer(args.norm))
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                     padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)


def _resnet_norm_layer(norm):
    norm = str(norm).lower()

    if norm in ('batch_norm', 'batchnorm', 'bn'):
        return nn.BatchNorm2d

    if norm in ('group_norm', 'groupnorm', 'gn'):
        def group_norm(num_channels):
            num_groups = min(32, num_channels)
            while num_channels % num_groups != 0:
                num_groups -= 1
            return nn.GroupNorm(num_groups, num_channels)
        return group_norm

    if norm in ('layer_norm', 'layernorm', 'ln'):
        return lambda num_channels: nn.GroupNorm(1, num_channels)

    if norm in ('none', 'identity'):
        return lambda num_channels: nn.Identity()

    raise ValueError(f'Unrecognized normalization layer: {norm}')
