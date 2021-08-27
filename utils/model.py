#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The model module
    ======================

    Use it to define, load and restore a model.
"""

import sys
import os
import logging
import time
import copy
import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class Net(nn.Module):
    """
    The Net class is used to generate a network.
    The class contains different useful layers.
    """
    def __init__(self, no_of_classes: int, use_amp: bool):
        """
        Constructor of the Net class.
        :param no_of_classes: The number of classes wanted at the
                              output of the network.
        :param use_amp: Whether to use Automatic Mixed Precision.
        """
        super(Net, self).__init__()
        self.amp = use_amp
        self.dilated_block1 = self.dilated_block(3, 32)
        self.dilated_block2 = self.dilated_block(32, 64)
        self.dilated_block3 = self.dilated_block(64, 128)
        self.dilated_block4 = self.dilated_block(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_block1 = self.conv_block(256, 128)
        self.conv_block2 = self.conv_block(256, 64)
        self.conv_block3 = self.conv_block(128, 32)
        self.last_conv = nn.Conv2d(64, no_of_classes, 3,
                                   stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def dilated_block(input_size: int, output_size: int):
        """
        Define a dilated block.
        It consists in 6 successive convolutions with the dilatations
        rates [1, 2, 4, 8, 16].
        :param input_size: The size of the input tensor.
        :param output_size: The size of the output tensor.
        :return: The sequence of the convolutions.
        """
        modules = []
        modules.append(nn.Conv2d(input_size, output_size, 3, stride=1,
                                 dilation=1, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(output_size,
                                      track_running_stats=False))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout(p=0.4))
        for i in [2, 4, 8, 16]:
            modules.append(nn.Conv2d(output_size, output_size, 3,
                                     stride=1, dilation=i, padding=i,
                                     bias=False))
            modules.append(nn.BatchNorm2d(output_size,
                                          track_running_stats=False))
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Dropout(p=0.4))
        return nn.Sequential(*modules)

    @staticmethod
    def conv_block(input_size: int, output_size: int):
        """
        Define a convolutional block.
        It consists in a convolution followed by an upsampling layer.
        :param input_size: The size of the input tensor.
        :param output_size: The size of the output tensor.
        :return: The sequence of the convolutions.
        """
        return nn.Sequential(
            nn.Conv2d(input_size, output_size, 3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_size, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            # Does the upsampling.
            nn.ConvTranspose2d(output_size, output_size,
                               2, stride=2, bias=False),
            nn.BatchNorm2d(output_size, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

    def forward(self, x):
        """
        Define the forward step of the network.
        It consists in 4 successive dilated blocks followed by 3
        convolutional blocks and a final convolution.
        :param x: The input tensor.
        :return x: The output tensor.
        """
        with autocast(enabled=self.amp):
            x = self.dilated_block1(x)
            out_block1 = x
            x = self.dilated_block2(self.pool(x))
            out_block2 = x
            x = self.dilated_block3(self.pool(x))
            out_block3 = x
            x = self.dilated_block4(self.pool(x))
            x = self.conv_block1(x)
            x = torch.cat([x, out_block3], dim=1)
            x = self.conv_block2(x)
            x = torch.cat([x, out_block2], dim=1)
            x = self.conv_block3(x)
            x = torch.cat([x, out_block1], dim=1)
            x = self.last_conv(x)
            x = self.softmax(x)
        return x


def weights_init(model):
    """
    Initialize the model weights.
    :param model: The model.
    """
    if isinstance(model, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(model.weight.data)


def load_network(no_of_classes: int, use_amp: bool, ex):
    """
    Load the network for the experiment.
    :param no_of_classes: The number of classes involved in the experiment.
    :param use_amp: Whether to use Automatic Mixed Precision.
    :param ex: The Sacred object to log information.
    :return net: The loaded network.
    :return last_layer: The last activation function to apply.
    """
    # Define the network.
    net = Net(no_of_classes, use_amp)
    # Allow parallel running if more than 1 gpu available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Running on %s', device)
    if torch.cuda.device_count() > 1:
        logging.info("Let's use %d GPUs", torch.cuda.device_count())
        net = nn.DataParallel(net)
        ex.log_scalar('gpus.number', torch.cuda.device_count())
    return net.to(device)


def restore_model(net, optimizer, scaler, log_path: str, model_path: str):
    """
    Load the model weights.
    :param net: The loaded model.
    :param optimizer: The loaded optimizer.
    :param scaler: The scaler used for AMP.
    :param log_path: The directory containing the model to restore.
    :param model_path: The name of the model to restore.
    :return checkpoint: The loaded checkpoint.
    :return net: The restored model.
    :return optimizer: The restored optimizer.
    :return scaler: The restored scaler.
    """
    starting_time = time.time()
    if not os.path.isfile(os.path.join(log_path, model_path)):
        logging.error('No model found at %s',
                      os.path.join(log_path, model_path))
        sys.exit()
    else:
        if torch.cuda.is_available():
            checkpoint = torch.load(os.path.join(log_path, model_path))
        else:
            checkpoint = torch.load(os.path.join(log_path, model_path),
                                    map_location=torch.device('cpu'))
        loaded_checkpoint = {}
        if torch.cuda.device_count() > 1:
            for key in checkpoint["state_dict"].keys():
                if 'module' not in key:
                    loaded_checkpoint['module.'+key] = checkpoint["state_dict"][key]
                else:
                    loaded_checkpoint = checkpoint["state_dict"]
        else:
            for key in checkpoint["state_dict"].keys():
                loaded_checkpoint[key.replace("module.", "")] = checkpoint["state_dict"][key]
        net.load_state_dict(loaded_checkpoint)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logging.info('Loaded checkpoint %s (epoch %d) in %1.5fs',
                     model_path, checkpoint['epoch'], (time.time() - starting_time))
        return checkpoint, net, optimizer, scaler


def save_model(epoch: int, model, loss: float, optimizer, scaler, filename: str):
    """
    Save the given model.
    :param epoch: The current epoch.
    :param model: The model state dict to save.
    :param loss: The loss of the current epoch.
    :param optimizer: The optimizer state dict.
    :param scaler: The scaler used for AMP.
    :param filename: The name of the model file.
    """
    model_params = {'epoch': epoch,
                    'state_dict': copy.deepcopy(model),
                    'best_loss': loss,
                    'optimizer': copy.deepcopy(optimizer),
                    'scaler': scaler}
    torch.save(model_params, filename)
