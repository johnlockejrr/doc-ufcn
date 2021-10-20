#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import cv2
import numpy as np
import torch

from doc_ufcn import image, model, prediction

logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.DEBUG,
)


class DocUFCN:
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
        assert self.no_of_classes > 0
        assert isinstance(self.no_of_classes, int)
        self.model_input_size = model_input_size
        self.device = device

    def load(self, model_path, mean, std):
        """
        Load a trained model.
        :param model_path: Path to the model.
        :param mean: The mean value to use to normalize the input image.
        :param std: The std value to use to normalize the input image.
        :return net: The loaded model.
        """
        net = model.DocUFCNModel(self.no_of_classes)
        net.to(self.device)
        # Restore the model weights.
        checkpoint = torch.load(
            model_path, map_location=self.device
        )  # TODO Check that the model exists
        loaded_checkpoint = {}
        for key in checkpoint["state_dict"].keys():
            loaded_checkpoint[key.replace("module.", "")] = checkpoint["state_dict"][
                key
            ]
        net.load_state_dict(loaded_checkpoint, strict=False)
        logging.debug(f"Loaded model {model_path}")
        self.net = net
        self.mean, self.std = mean, std

    def predict(
        self,
        input_image,
        min_cc=50,
        raw_output=False,
        mask_output=False,
        overlap_output=False,
    ):
        self.net.eval()
        input_size = (input_image.shape[0], input_image.shape[1])

        input_image = np.asarray(input_image)
        if len(input_image.shape) < 3:
            input_image = cv2.cvtColor(
                input_image, cv2.COLOR_GRAY2RGB
            )  # TODO Image should be in RGB

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

        # Remove the small connected components.
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

        return predicted_polygons, pred, mask * 255 / np.max(mask), overlap
