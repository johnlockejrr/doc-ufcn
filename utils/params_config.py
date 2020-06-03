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


class TrainingParams(BaseParams):
    """
    This is a class for the configuration parameters of the training step.
    """
    def __init__(self, **kwargs):
        """
        Constructor of the TrainingParams class.
        :param model_path: The path to the trained model.
        :param train_frame_path: The directory containing the training images.
        :param train_mask_path: The directory containing the labels of the
                                training images.
        :param val_frame_path: The directory containing the validation images.
        :param val_mask_path: The directory containing the labels of the
                              validation images.
        :param classes_file: The text file containing the color codes of
                             the different classes.
        """
        self.model_path = kwargs.get('model_path', 'model.pth')
        self.train_frame_path = kwargs.get('train_frame_path',
                                           './data/train/images/')
        self.train_mask_path = kwargs.get('train_mask_path',
                                          './data/train/labels/')
        self.val_frame_path = kwargs.get('val_frame_path',
                                         './data/val/images/')
        self.val_mask_path = kwargs.get('val_mask_path',
                                        './data/val/labels/')
        self.classes_file = kwargs.get('classes_dir', './data/classes.txt')


class PredictionParams(BaseParams):
    """
    This is a class for the configuration parameters of the prediction step.
    """
    def __init__(self, **kwargs):
        """
        Constructor of the PredictionParams class.
        :param model_path: The path to the trained model.
        :param test_frame_path: The directory containing the testing images.
        :param predictions_path: The directory to save the predicted images.
        :param predictions_pp_path: The directory to save the postprocessed
                                    predictions images.
        :param classes_file: The text file containing the color codes of
                             the different classes.
        """
        self.model_path = kwargs.get('model_path', 'model.pth')
        self.test_frame_path = kwargs.get('test_frame_path',
                                          './data/test/images/')
        self.predictions_path = kwargs.get('predictions_path',
                                           'res/predictions/')
        self.predictions_pp_path = kwargs.get('predictions_pp_path',
                                              'res/predictions_pp/')
        self.classes_file = kwargs.get('classes_dir', './data/classes.txt')


class TestParams(BaseParams):
    """
    This is a class for the configuration parameters of the evaluation step.
    """
    def __init__(self, **kwargs):
        """
        Constructor of the TestParams class.
        :param model_path: The path to the trained model.
        :param test_frame_path: The directory containing the testing images.
        :param test_mask_path: The directory containing the labels of the
                               testing images.
        :param results_dir: The directory to save the results.
        :param classes_file: The text file containing the color codes of
                             the different classes.
        """
        self.model_path = kwargs.get('model_path', 'model.pth')
        self.test_frame_path = kwargs.get('test_frame_path',
                                          './data/test/images/')
        self.test_mask_path = kwargs.get('test_mask_path',
                                         './data/test/labels/')
        self.results_dir = kwargs.get('results_dir', 'res/')
        self.classes_file = kwargs.get('classes_dir', './data/classes.txt')
