import copy
import logging
import sys
import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch.nn import Module as NNModule

logger = logging.getLogger(__name__)


class DocUFCNModel(NNModule):
    """
    The DocUFCNModel class is used to generate the Doc-UFCN network.
    The class initializes different useful layers and defines
    the sequencing of the defined layers/blocks.
    """

    def __init__(self, no_of_classes, use_amp=False):
        """
        Constructor of the DocUFCNModel class.
        :param no_of_classes: The number of classes wanted at the
                              output of the network.
        :param use_amp: Whether to use Automatic Mixed Precision.
                        Disabled by default
        """
        super().__init__()
        self.amp = use_amp
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
            torch.nn.Conv2d(
                input_size, output_size, 3, stride=1, dilation=1, padding=1, bias=False
            )
        )
        modules.append(torch.nn.BatchNorm2d(output_size, track_running_stats=False))
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(torch.nn.Dropout(p=0.4))
        for i in [2, 4, 8, 16]:
            modules.append(
                torch.nn.Conv2d(
                    output_size,
                    output_size,
                    3,
                    stride=1,
                    dilation=i,
                    padding=i,
                    bias=False,
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
            torch.nn.Conv2d(
                input_size, output_size, 3, stride=1, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(output_size, track_running_stats=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.4),
            # Does the upsampling.
            torch.nn.ConvTranspose2d(output_size, output_size, 2, stride=2, bias=False),
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
        with autocast(enabled=self.amp):
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


def weights_init(model):
    """
    Initialize the model weights.
    :param model: The model.
    """
    if isinstance(model, torch.nn.Conv2d | torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(model.weight.data)


def load_network(no_of_classes: int, use_amp: bool):
    """
    Load the network for the experiment.
    :param no_of_classes: The number of classes involved in the experiment.
    :param use_amp: Whether to use Automatic Mixed Precision.
    :return net: The loaded network.
    :return last_layer: The last activation function to apply.
    """
    # Define the network.
    net = DocUFCNModel(no_of_classes, use_amp)
    # Allow parallel running if more than 1 gpu available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Running on %s", device)
    if torch.cuda.device_count() > 1:
        logger.info("Let's use %d GPUs", torch.cuda.device_count())
        net = torch.nn.DataParallel(net)
    return net.to(device)


def restore_model(
    net, optimizer, scaler, log_path: Path, model_path: Path, keep_last: bool = True
):
    """
    Load the model weights.
    :param net: The loaded model.
    :param optimizer: The loaded optimizer.
    :param scaler: The scaler used for AMP.
    :param log_path: The directory containing the model to restore.
    :param model_path: The name of the model to restore.
    :param keep_last: Retrieve the last layer weights.
    :return checkpoint: The loaded checkpoint.
    :return net: The restored model.
    :return optimizer: The restored optimizer.
    :return scaler: The restored scaler.
    """
    starting_time = time.time()
    if not (log_path / model_path).is_file():
        logger.error("No model found at %s", log_path / model_path)
        sys.exit()
    else:
        if torch.cuda.is_available():
            checkpoint = torch.load(log_path / model_path)
        else:
            checkpoint = torch.load(
                log_path / model_path, map_location=torch.device("cpu")
            )
        loaded_checkpoint = {}
        if torch.cuda.device_count() > 1:
            for key in checkpoint["state_dict"]:
                if "module" not in key:
                    loaded_checkpoint[f"module.{key}"] = checkpoint["state_dict"][key]
                else:
                    loaded_checkpoint = checkpoint["state_dict"]
        else:
            for key in checkpoint["state_dict"]:
                loaded_checkpoint[key.replace("module.", "")] = checkpoint[
                    "state_dict"
                ][key]

        if keep_last:
            net.load_state_dict(loaded_checkpoint)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None:
                scaler.load_state_dict(checkpoint["scaler"])
        else:
            loaded_checkpoint.pop("last_conv.weight")
            loaded_checkpoint.pop("last_conv.bias")
            net.load_state_dict(loaded_checkpoint, strict=False)

        logger.info(
            "Loaded checkpoint %s (epoch %d) in %1.5fs",
            model_path,
            checkpoint["epoch"],
            (time.time() - starting_time),
        )
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
    model_params = {
        "epoch": epoch,
        "state_dict": copy.deepcopy(model),
        "best_loss": loss,
        "optimizer": copy.deepcopy(optimizer),
        "scaler": scaler,
    }
    torch.save(model_params, filename)
