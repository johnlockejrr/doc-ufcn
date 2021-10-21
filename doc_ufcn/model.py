#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.nn import Module as NNModule


class DocUFCNModel(NNModule):
    """
    The DocUFCNModel class is used to generate the Doc-UFCN network.
    The class initializes different useful layers and defines
    the sequencing of the defined layers/blocks.
    """

    def __init__(self, no_of_classes):
        """
        Constructor of the DocUFCNModel class.
        :param no_of_classes: The number of classes wanted at the
                              output of the network.
        """
        super(DocUFCNModel, self).__init__()
        self.dilated_block1 = self.dilated_block(3, 32)
        self.dilated_block2 = self.dilated_block(32, 64)
        self.dilated_block3 = self.dilated_block(64, 128)
        self.dilated_block4 = self.dilated_block(128, 256)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv_block1 = self.conv_block(256, 128)
        self.conv_block2 = self.conv_block(256, 64)
        self.conv_block3 = self.conv_block(128, 32)
        self.last_conv = torch.nn.Conv2d(64, no_of_classes, 3, stride=1, padding=1)
        self.softmax = torch.nn.Softmax(dim=1)

    @staticmethod
    def dilated_block(input_size, output_size):
        """
        Define a dilated block.
        It consists in 6 successive convolutions with the dilations
        rates [1, 2, 4, 8, 16].
        :param input_size: The size of the input tensor.
        :param output_size: The size of the output tensor.
        :return: The sequence of the convolutions.
        """
        modules = []
        modules.append(
            torch.nn.Conv2d(input_size, output_size, 3, stride=1, dilation=1, padding=1)
        )
        modules.append(torch.nn.BatchNorm2d(output_size, track_running_stats=False))
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(torch.nn.Dropout(p=0.4))
        for i in [2, 4, 8, 16]:
            modules.append(
                torch.nn.Conv2d(
                    output_size, output_size, 3, stride=1, dilation=i, padding=i
                )
            )
            modules.append(torch.nn.BatchNorm2d(output_size, track_running_stats=False))
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Dropout(p=0.4))
        return torch.nn.Sequential(*modules)

    @staticmethod
    def conv_block(input_size, output_size):
        """
        Define a convolutional block.
        It consists in a convolution followed by an upsampling layer.
        :param input_size: The size of the input tensor.
        :param output_size: The size of the output tensor.
        :return: The sequence of the convolutions.
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(input_size, output_size, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(output_size, track_running_stats=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.4),
            # Does the upsampling.
            torch.nn.ConvTranspose2d(output_size, output_size, 2, stride=2),
            torch.nn.BatchNorm2d(output_size, track_running_stats=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.4),
        )

    def forward(self, input_tensor):
        """
        Define the forward step of the network.
        It consists in 4 successive dilated blocks followed by 3
        convolutional blocks, a final convolution and a softmax layer.
        :param input_tensor: The input tensor.
        :return: The output tensor.
        """
        tensor = self.dilated_block1(input_tensor)
        out_block1 = tensor
        tensor = self.dilated_block2(self.pool(tensor))
        out_block2 = tensor
        tensor = self.dilated_block3(self.pool(tensor))
        out_block3 = tensor
        tensor = self.dilated_block4(self.pool(tensor))
        tensor = self.conv_block1(tensor)
        tensor = torch.cat([tensor, out_block3], dim=1)
        tensor = self.conv_block2(tensor)
        tensor = torch.cat([tensor, out_block2], dim=1)
        tensor = self.conv_block3(tensor)
        tensor = torch.cat([tensor, out_block1], dim=1)
        output_tensor = self.last_conv(tensor)
        return self.softmax(output_tensor)
