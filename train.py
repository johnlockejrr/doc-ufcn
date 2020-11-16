#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The train module
    ======================

    Use it to train a network.

    :example:

    >>> python train.py with utils/training_config.json
"""

import os
import logging
import time
import numpy as np
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import MongoObserver
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import utils.model as m
from utils.data import load_data
from utils.params_config import TrainingParams
from utils.metrics import PixelMetrics
from utils.utils import get_classes_colors, display_training

ex = Experiment('U-FCN')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
#ex.observers.append(MongoObserver(
#    url='mongodb://user:password@omniboard.vpn/dbname'))


@ex.config
def default_config():
    """
    Define the default configuration for the experiment.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = TrainingParams().to_dict()
    batch_size = 4
    no_of_epochs = 2
    img_size = 384
    learning_rate = 5e-3
    experiment_name = 'ufcn'
    log_path = 'runs/'+experiment_name.lower().replace(' ', '_').replace('-', '_')
    tb_path = 'events/'
    no_of_classes = 2
    restore_model = False
    classes_names = ["Background", "Text_line"]
    normalization_params = {"mean": [0, 0, 0], "std": [1, 1, 1]}


class diceloss(nn.Module):
    def __init__(self):
        super(diceloss, self).__init__()

    def forward(self, pred, target):

       label = nn.functional.one_hot(target, num_classes=2).permute(0,3,1,2).contiguous()

       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = label.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


@ex.capture
def initialize_experiment(classes: list, params: TrainingParams,
                          img_size: int, no_of_classes: int, device: str,
                          normalization_params: dict, batch_size: int,
                          learning_rate: float):
    """
    Initialize the experiment.
    Load the data into batches and load the network.
    :param classes: The color codes of the different classes.
    :param params: The dictionnary containing the parameters
                   of the experiment.
    :param img_size: The size in which the images will be resized.
    :param no_of_classes: The number of classes involved in the experiment.
    :param device: The device used to run the experiment.
    :param normalization_params: The mean values and standard deviations used
                                 to normalize the images.
    :param batch_size: The size of the batch to use during training.
    :param learning_rate: The learning_rate to start the training.
    :return train_loader: The loader containing the pre-processed images
                          for training.
    :return val_loader: The loader containing the pre-processed images
                        for validation.
    :return net: The loaded network.
    :return last_layer: The last activation function to apply.
    :return criterion: The loss to use during training.
    :return optimizer: The optimizer to use during training.
    """
    params = TrainingParams.from_dict(params)

    train_loader = load_data(params.train_frame_path, img_size,
                             normalization_params['mean'],
                             normalization_params['std'], classes,
                             mask_path=params.train_mask_path,
                             batch_size=batch_size)

    val_loader = load_data(params.val_frame_path, img_size,
                           normalization_params['mean'],
                           normalization_params['std'], classes,
                           mask_path=params.val_mask_path,
                           batch_size=batch_size)

    net, last_layer = m.load_network(no_of_classes, device, ex)
    criterion = diceloss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # Generate model graph.
    m.generate_model_graph(net, img_size)
    return train_loader, val_loader, net, last_layer, criterion, optimizer


@ex.capture
def run_one_epoch(loader, net, criterion, optimizer, writer, epoch: int,
                  device: str, normalization_params: dict, step: str,
                  no_of_classes: int, classes_names: list):
    """
    Train for one epoch.
    :param loader: The loader containing the pre-processed images.
    :param net: The network to use.
    :param criterion: The loss to use.
    :param optimizer: The optimizer to use.
    :param writer: The Tensorboard writer to log the information.
    :param epoch: The current epoch.
    :param device: The device used to run the experiment.
    :param normalization_params: The mean values and standard deviations used
                                 to normalize the images.
    :param step: A string indicating whether it's a training or a validation
                 step.
    :param no_of_classes: The number of different classes.
    :param classes_names: The names of the classes involved in the experiment.
    :return running_loss: The loss of the epoch.
    :return miou: The mean Interesection-over-Union for the epoch.
    :return criterion: The updated criterion.
    :return optimizer: The updated optimizer.
    """
    running_loss = []
    miou = {channel: [] for channel in classes_names}
    for index, data in enumerate(tqdm(loader, desc=step), 0):
        image, label = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        output = net[0](image.float())

        loss = criterion(output, label.long())
        running_loss.append(loss.item())

        if step == "Training":
            loss.backward()
            optimizer.step()
            # Display prediction images in Tensorboard all 200 mini-batches.
            if index % 200 == 199:
                display_training(output, image, label, writer,
                                 epoch, normalization_params)

        for pred in range(output.shape[0]):
            current_pred = output[pred, :, :, :].cpu().detach().numpy()
            current_pred = np.argmax(current_pred, axis=0)
            current_label = label[pred, :, :].cpu().numpy()
            metrics = PixelMetrics(current_pred, current_label, classes_names)
            metrics.compute_confusion_matrix()
            ious = metrics.compute_iou()
            for channel in classes_names:
                miou[channel].append(ious[channel])

    for channel in classes_names:
        miou[channel] = np.mean(miou[channel]) 

    if step == "Training":
        return np.mean(running_loss), miou, criterion, optimizer
    else:
        return np.mean(running_loss), miou


@ex.capture
def log_metrics(epoch: int, loss: float, iou: float, writer, step: str,
                classes_names: list):
    """
    Log the results into Omniboard and Tensorboard.
    :param epoch: The current epoch.
    :param loss: The loss of the current epoch.
    :param iou: The mean Interesection-over-Union for the epoch.
    :param writer: The Tensorboard writer to log the information.
    :param step: A string indicating whether it's a training or a validation
                 step.
    :param learning_rate: The current learning_rate.
    """
    print('[Epoch %d] %s mean_loss: %.3f'
          % (epoch+1, step, loss))
    writer.add_scalar(step+'_loss', loss, epoch)
    ex.log_scalar(step.lower()+'.loss', loss, epoch)
    for channel in classes_names:
        print('  mean_iou %s: %.3f' % (channel.lower(), iou[channel]))   
        writer.add_scalar(step+'_mean_iou_'+channel.lower(),
                          iou[channel], epoch)
        ex.log_scalar(step.lower()+'.mean_iou.'+channel.lower(),
                      iou[channel], epoch)


def weights_init(model):
    """
    Initialize the model weights.
    :param model: The model.
    """
    if isinstance(model, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(model.weight.data)


@ex.automain
def run(params: TrainingParams, log_path: str, restore_model: bool,
        tb_path: str, no_of_epochs: int):
    """
    Main function. Run the training.
    :param params: The dictionnary containing the parameters
                   of the experiment.
    :param log_path: The directory to log the training information for
                     the current experiment.
    :param restore_model: A boolean indicating whether to resume a training.
    :param tb_path: The directory containing the Tensorboard log files.
    :param no_of_epochs: The number of epochs to train.
    """
    params = TrainingParams.from_dict(params)
    # Get the possible colors.
    colors = get_classes_colors(params.classes_file)

    train_loader, val_loader, net, softmax, criterion, \
        optimizer = initialize_experiment(colors)
    net.apply(weights_init)

    saved_epoch = 0
    if restore_model:
        checkpoint, net, optimizer = m.restore_model_parameters(
            net, optimizer, log_path, params.model_path)
        saved_epoch = checkpoint['epoch']-1
        optimizer = optim.Adam(net.parameters(), lr=5e-3)
    best_loss = 10e5

    writer = SummaryWriter(os.path.join(log_path, tb_path))
    # Train the model.
    logging.info('Starting training')
    starting_time = time.time()
    for epoch in range(no_of_epochs):
        current_epoch = epoch + saved_epoch
        # Run training.
        net.train()
        loss, iou, criterion, optimizer = run_one_epoch(
            train_loader, [net, softmax], criterion, optimizer, writer,
            current_epoch, step="Training")
        log_metrics(current_epoch, loss, iou, writer, step="Training")
        with torch.no_grad():
            # Run evaluation.
            net.eval()
            loss, iou = run_one_epoch(val_loader, [net, softmax], criterion, optimizer,
                                      writer, current_epoch, step="Validation")
            log_metrics(current_epoch, loss, iou, writer, step="Validation")
            # Keep best model.
            if loss < best_loss:
                best_loss = loss
                m.save_model(current_epoch+1, net.module.state_dict(), loss,
                             optimizer.state_dict(), params.model_path, log_path)
                logging.info('Best model (epoch %d) saved', current_epoch+1)
    # Save last model.
    if restore_model:
        m.save_model(current_epoch+1, net.module.state_dict(), loss,
                     optimizer.state_dict(), 'last_2_'+params.model_path,
                     log_path)
    else:
        m.save_model(current_epoch+1, net.module.state_dict(), loss,
                     optimizer.state_dict(), 'last_'+params.model_path,
                     log_path)
    logging.info('Last model (epoch %d) saved', current_epoch+1)

    end = time.gmtime(time.time() - starting_time)
    logging.info('Finished training in %2d:%2d:%2d',
                 end.tm_hour, end.tm_min, end.tm_sec)
