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
from torchviz import make_dot


class Net(nn.Module):
    """
    The Net class is used to generate a network.
    The class contains different useful layers.
    """
    def __init__(self, no_of_classes: int):
        """
        Constructor of the Net class.
        :param no_of_classes: The number of classes wanted at the
                              output of the network.
        """
        super(Net, self).__init__()
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
        return x


def generate_model_graph(net, img_size: int):
    """
    Generate the graph of the model.
    :param net: The network we want the graph.
    :param img_size: The size in which the images are resized.
    """
    fake_input = torch.zeros(1, 3, img_size, img_size, dtype=torch.float,
                             requires_grad=False)
    fake_output = net(fake_input)
    graph = make_dot(fake_output)
    graph.format = 'svg'
    graph.render(filename='graph')


def load_network(no_of_classes: int, device: str, ex):
    """
    Load the network for the experiment.
    :param no_of_classes: The number of classes involved in the experiment.
    :param device: The device used to run the experiment.
    :return net: The loaded network.
    :return last_layer: The last activation function to apply.
    """
    # Define the network.
    net = Net(no_of_classes)
    # Allow parallel running if more than 1 gpu available.
    logging.info('Running on %s', device)
    if torch.cuda.device_count() > 1:
        logging.info("Let's use %d GPUs", torch.cuda.device_count())
        net = nn.DataParallel(net)
        ex.log_scalar('gpus.number', torch.cuda.device_count())
    net.to(device)
    last_layer = nn.Softmax(dim=1)
    return net, last_layer


def restore_model(net, log_path: str, model_path: str):
    """
    Load the model weights.
    :param net: The loaded model.
    :param log_path: The directory containing the model to restore.
    :param model_path: The name of the model to restore.
    :return net: The restored model.
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
        net_state_dict = dict(
            ('module.'+key, value) for (key, value)
            in checkpoint['state_dict'].items())
        logging.info('Loaded epoch %d', checkpoint['epoch'])
        net.load_state_dict(net_state_dict)
        logging.info('Loaded model in %1.5fs', (time.time() - starting_time))
        return net


def restore_model_parameters(net, optimizer, log_path: str, model_path: str):
    """
    Load the model weights to resume a training.
    :param net: The loaded model.
    :param optimizer: The optimizer used during training.
    :param log_path: The directory containing the model to restore.
    :param model_path: The name of the model to restore.
    :return checkpoint: The loaded checkpoint.
    :return net: The restored model.
    :return optimizer: The restored optimizer.
    """
    # Restore model to continue training
    if os.path.isfile(os.path.join(log_path, 'last_'+model_path)):
        checkpoint = torch.load(os.path.join(log_path, 'last_'+model_path))
        net_state_dict = dict(
            ('module.'+key, value) for (key, value)
            in checkpoint['state_dict'].items())
        net.load_state_dict(net_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info('Loaded checkpoint %s (epoch %d)',
                     'last_'+model_path, checkpoint['epoch'])
        return checkpoint, net, optimizer
    else:
        logging.error('No checkpoint found at %s',
                      os.path.join(log_path, 'last_'+model_path))
        sys.exit()


def save_model(epoch: int, model, loss: float, optimizer, filename: str,
               log_path: str):
    """
    Save the given model.
    :param epoch: The current epoch.
    :param model: The model state dict to save.
    :param loss: The loss of the current epoch.
    :param optimizer: The optimizer state dict.
    :param filename: The name of the model file.
    :param log_path: The directory of the training information for
                     the current experiment.
    """
    model_params = {'epoch': epoch,
                    'state_dict': copy.deepcopy(model),
                    'best_loss': loss,
                    'optimizer': copy.deepcopy(optimizer)}
    torch.save(model_params, os.path.join(log_path, filename))
