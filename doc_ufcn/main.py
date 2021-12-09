#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import cv2
import numpy as np
import torch

from doc_ufcn import image, model, prediction


class DocUFCN:
    """
    The DocUFCN class is used to apply the Doc-UFCN model.
    The class initializes useful parameters: number of classes,
    model input size and the device.
    """

    def __init__(self, no_of_classes, model_input_size, device):
        """
        Constructor of the DocUFCN class.
        :param no_of_classes: The number of classes wanted at the
                              output of the network.
        :param model_input_size: The size of the model input.
        :param device: The device to use.
        """
        super(DocUFCN, self).__init__()
        self.no_of_classes = no_of_classes
        assert isinstance(
            self.no_of_classes, int
        ), "Number of classes must be an integer"
        assert self.no_of_classes > 0, "Number of classes must be positive"
        self.model_input_size = model_input_size
        assert isinstance(
            self.model_input_size, int
        ), "Model input size must be an integer"
        assert self.model_input_size > 0, "Model input size must be positive"
        self.device = device

    def load(self, model_path, mean, std, mode="eval"):
        """
        Load a trained model.
        :param model_path: Path to the model.
        :param mean: The mean value to use to normalize the input image.
        :param std: The std value to use to normalize the input image.
        :param mode: The mode to load the model (train or eval).
        """
        net = model.DocUFCNModel(self.no_of_classes)
        net.to(self.device)

        if mode == "train":
            net.train()
        elif mode == "eval":
            net.eval()
        else:
            raise Exception("Unsupported mode")

        # Restore the model weights.
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        loaded_checkpoint = {}
        for key in checkpoint["state_dict"].keys():
            loaded_checkpoint[key.replace("module.", "")] = checkpoint["state_dict"][
                key
            ]
        net.load_state_dict(loaded_checkpoint, strict=False)
        logging.debug(f"Loaded model {model_path}")
        self.net = net
        self.mean, self.std = mean, std
        assert isinstance(
            mean, list
        ), "mean must be a list of 3 integers (RGB) between 0 and 255"
        assert (
            len(mean) == 3
        ), "mean must be a list of 3 integers (RGB) between 0 and 255"
        assert all(
            isinstance(element, int) and element >= 0 and element <= 255
            for element in mean
        ), "mean must be a list of 3 integers (RGB) between 0 and 255"
        assert isinstance(
            std, list
        ), "std must be a list of 3 integers (RGB) between 0 and 255"
        assert len(std) == 3, "std must be a list of 3 integers (RGB) between 0 and 255"
        assert all(
            isinstance(element, int) and element >= 0 and element <= 255
            for element in std
        ), "std must be a list of 3 integers (RGB) between 0 and 255"

    def predict(
        self,
        input_image,
        min_cc=50,
        raw_output=False,
        mask_output=False,
        overlap_output=False,
    ):
        """
        Run prediction on an input image.
        :param input_image: The image to predict.
        :param min_cc: The threshold to remove small connected components.
        :param raw_output: Return the raw probabilities.
        :param mask_output: Return a mask with the detected objects.
        :param overlap_output: Return the detected objects drawn over the input image.
        """
        assert isinstance(
            input_image, np.ndarray
        ), "Input image must be an np.array in RGB"
        input_size = (input_image.shape[0], input_image.shape[1])
        input_image = np.asarray(input_image)
        if len(input_image.shape) < 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)

        # Preprocess the input image.
        input_tensor, padding = image.preprocess_image(
            input_image, self.model_input_size, self.mean, self.std
        )
        logging.debug("Image pre-processed")

        # Run the prediction.
        with torch.no_grad():
            pred = self.net(input_tensor.float().to(self.device))
            pred = pred[0].cpu().detach().numpy()
            # Get contours of the predicted objects.
            predicted_polygons = prediction.get_predicted_polygons(
                pred, self.no_of_classes
            )
            logging.info("Image processed")

        # Remove the small connected components.
        assert isinstance(min_cc, int), "min_cc must be a positive integer"
        assert min_cc > 0, "min_cc must be a positive integer"
        if min_cc > 0:
            for channel in range(1, self.no_of_classes):
                predicted_polygons[channel] = [
                    contour
                    for contour in predicted_polygons[channel]
                    if cv2.contourArea(contour["polygon"]) > min_cc
                ]

        # Resize the polygons.
        resized_predicted_polygons = prediction.resize_predicted_polygons(
            predicted_polygons, input_size, self.model_input_size, padding
        )

        # Generate the mask images if requested.
        mask = (
            prediction.get_prediction_image(resized_predicted_polygons, input_size)
            if mask_output
            else None
        )
        overlap = (
            prediction.get_prediction_image(
                resized_predicted_polygons, input_size, input_image
            )
            if overlap_output
            else None
        )

        if not raw_output:
            pred = None

        if mask is not None:
            return predicted_polygons, pred, mask * 255 / np.max(mask), overlap
        return predicted_polygons, pred, mask, overlap
