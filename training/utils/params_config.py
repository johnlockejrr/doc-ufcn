#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The params_config module
    ======================

    Use it to define the configuration parameters to use.
"""


class BaseParams:
    """
    This is a global class for the configuration parameters.
    """
    def to_dict(self):
        """
        Maps the class attributes to a dictionary.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, dictionary: dict):
        """
        Maps a dictionary to the class attributes.
        :param dictionary: The dictionary to map.
        :return result: The object with the attributes.
        """
        result = cls()
        keys = result.to_dict().keys()
        for key, value in dictionary.items():
            assert key in keys
            setattr(result, key, value)
        return result


class Params(BaseParams):
    """
    This is a class for the configuration parameters.
    :param mean: Path to the file containing the mean values of training set.
    :param str: Path to the file containing the standard deviation values of training set.
    :param model_path: Path to store the obtained model.
    :param train_image_path: Path to the directory containing training images.
    :param train_mask_path: Path to the directory containing training masks. 
    :param val_image_path: Path to the directory containing validation images. 
    :param val_mask_path: Path to the directory containing validation masks. 
    :param test_image_path: Path to the directory containing testing images. 
    :param train_image_path: Path to the directory containing training images.
    :param classes_files: File containing the color codes of the classes involved
                          in the experiment.
    ;param prediction_path: Path to the directory to save the predictions. 
    """
    def __init__(self, **kwargs):

        self.mean = kwargs.get('mean', 'mean')
        self.std = kwargs.get('std', 'std')
        self.model_path = kwargs.get('model_path', 'model.pth')
        self.prediction_path = kwargs.get('prediction_path', 'prediction')
        self.evaluation_path = kwargs.get('evaluation_path', 'results')
