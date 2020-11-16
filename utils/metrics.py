#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The metrics module
    ======================

    Use it to compute different metrics from a predicted image.
    Available metrics:
        - Pixel IoU
        - Pixel Precision
        - Pixel Recall
        - Pixel F-score
        - Object Precision
        - Object Recall
        - Object F-score
        - Object AP
"""

import cv2
import numpy as np
from shapely.geometry import Polygon


def get_coords(image: np.ndarray, channel: int) -> list:
    """
    Get the coordinates of the objects of a given class in an image.
    :param image: The image to get objects' coordinates.
    :param channel: The class of the objects to get the coordinates.
    :return coords: The objects coordinates.
    """
    mask = np.uint8(image == channel)
    coords, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    return coords


def get_mean_point(contour: list) -> np.array:
    """
    Get the centroid of a contour.
    :param contour: The contour to get the centroid.
    :return: The centroid of the given contour.
    """
    moments = cv2.moments(contour)
    center_x = int(moments["m10"] / moments["m00"]) \
                   if moments["m00"] != 0 else 0
    center_y = int(moments["m01"] / moments["m00"]) \
                   if moments["m00"] != 0 else 0
    return np.asarray([center_x, center_y])


def compute_confidence(region: np.array, probs: np.array) -> float:
    """
    Get the mean confidence score of the model for a given region.
    :param region: The region to compute the confidence score.
    :param probs: The probability map.
    :return: The mean confidence score for the region.
    """
    mask = np.zeros(probs.shape)
    cv2.drawContours(mask, [region], 0, 1, -1)
    confidence = np.sum(mask*probs)/np.sum(mask)
    return round(confidence, 2)


class PixelMetrics():
    """
    The PixelMetrics class is used to compute the various metrics at pixel
    level. It requires a prediction and a label image.
    It also computes the mean metrics values at the end of the evaluation
    stage.
    """
    def __init__(self, prediction: np.array, label: np.array, classes: list):
        """
        Constructor of the PixelMetrics class.
        :param prediction: The current prediction.
        :param label: The corresponding label.
        :param classes: The different classes names involved in the experiment.
        """
        self.prediction = prediction
        self.label = label
        self.classes = classes

    def compute_confusion_matrix(self) -> np.array:
        """
        Get the confusion matrix between the prediction and the given label.
        :return confusion_matrix: The computed confusion matrix.
        """
        size = len(self.classes)
        confusion_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                bin_label = self.label == i
                bin_pred = self.prediction == j
                confusion_matrix[j, i] = (bin_pred*bin_label).sum()
        self.confusion_matrix = confusion_matrix
        return confusion_matrix

    def compute_iou(self) -> dict:
        """
        Get the Intersection-over-Union values between the prediction and the
        given label. It returns one value by class.
        :return ious: The computed Intersection-over-Union values by channel.
        """
        ious = {channel: 0 for channel in self.classes}
        for index, channel in enumerate(self.classes):
            true_positives = self.confusion_matrix[index, index]
            tpfn = np.sum(self.confusion_matrix[:, index])
            tpfp = np.sum(self.confusion_matrix[index, :])
            ious[channel] = true_positives / (tpfn + tpfp - true_positives)
        return ious

    def compute_precision(self) -> dict:
        """
        Get the precision between the prediction and the given label.
        It returns one value by class.
        :return precisions: The computed precisions by channel.
        """
        precisions = {channel: 0 for channel in self.classes}
        for index, channel in enumerate(self.classes):
            true_positives = self.confusion_matrix[index, index]
            tpfp = np.sum(self.confusion_matrix[index, :])
            precisions[channel] = true_positives/tpfp if tpfp != 0 else np.nan
        self.precision = precisions
        return precisions

    def compute_recall(self):
        """
        Get the recall between the prediction and the given label.
        It returns one value by class.
        :return recalls: The computed recalls by channel.
        """
        recalls = {channel: 0 for channel in self.classes}
        for index, channel in enumerate(self.classes):
            true_positives = self.confusion_matrix[index, index]
            tpfn = np.sum(self.confusion_matrix[:, index])
            recalls[channel] = true_positives / tpfn if tpfn != 0 else np.nan
        self.recall = recalls
        return recalls

    def compute_fscore(self) -> dict:
        """
        Get the F-score between the prediction and the given label.
        It returns one value by class.
        :return fscores: The computed F-scores by channel.
        """
        fscores = {channel: 0 for channel in self.classes}
        for channel in self.classes:
            ch_precision = self.precision[channel]
            ch_recall = self.recall[channel]
            fscores[channel] = 2 * (ch_precision * ch_recall) / \
                (ch_precision + ch_recall)
        return fscores

    def update_results(self, results: dict, conf_matrix: np.array):
        """
        Update the global results by adding the current metrics values.
        :param results: The dictionary containing the global results.
        :param conf_matrix: The confusion matrix to update with new results.
        :return results: The updated dictionary.
        :return conf_matrix: The updated confusion matrix.
        """
        conf_matrix += self.compute_confusion_matrix()
        for metric in results:
            value = getattr(self, 'compute_'+metric)()
            for channel in self.classes:
                results[metric][channel].append(value[channel])
        return results, conf_matrix

    def get_mean_results(self, results: dict) -> dict:
        """
        Get the mean metrics values for all the test set.
        :param results: The dictionary containing all the computed results.
        :return results: The dictionary containing the mean computed values.
        """
        for metric in results:
            if isinstance(results[metric], dict):
                for channel in self.classes:
                    res = [one_result
                           for one_result in results[metric][channel]
                           if str(one_result) != 'nan']
                    results[metric][channel] = np.mean(res)
        return results


class ObjectMetrics():
    """
    The ObjectMetrics class is used to compute the various metrics at object
    level. It requires a prediction and a label image.
    It also computes the mean metrics values at the end of the evaluation
    stage.
    """
    def __init__(self, prediction: np.array, label: np.array,
                 classes: list, probs: np.ndarray):
        """
        Constructor of the ObjectMetrics class.
        :param prediction: The current prediction.
        :param label: The corresponding label.
        :param classes: The different classes names involved in the experiment.
        :param probs: The probabilities obtained at the end of the network.
        """
        self.prediction = prediction
        self.label = label
        self.classes = classes
        self.probabilities = probs
        self.positive_examples = {channel: 0 for channel in self.classes[1:]}
        self.rank_scores = self.compute_rank_scores()

    def __compute_iou_from_contours(self, poly1: np.array,
                                    poly2: np.array) -> float:
        """
        Get the Intersection-over-Union value between two contours.
        :param poly1: The first contour used to compute the
                      Intersection-over-Union.
        :param poly2: The second contour used to compute the
                      Intersection-over-Union.
        :return: The computed Intersection-over-Union value between the two
                 contours.
        """
        poly1 = Polygon(poly1).buffer(0)
        poly2 = Polygon(poly2).buffer(0)
        intersection = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - intersection
        return intersection / union

    def __get_ious_by_channel(self, coords_prediction: list,
                              coords_label: list) -> dict:
        """
        Get the Intersection-over-Union values of the object of a given
        class.
        :param coords_prediction: The coordinates of the predicted objects.
        :param coords_label: The coordinates of the ground truth objects.
        :return ious: The computed Intersection-over-Union between the
                      predicted and ground truth objects.
        """
        ious = {key: 0 for key in range(len(coords_prediction))}
        size = np.asarray(self.prediction.shape[0:1])
        for coord_label in coords_label:
            coord_label = np.asarray([element[0] for element in coord_label])
            mean_label = get_mean_point(coord_label)
            best_iou = 0
            best_prediction = None
            for index_pred, coord_pred in enumerate(coords_prediction):
                coord_pred = np.asarray([element[0] for element in coord_pred])
                mean_prediction = get_mean_point(coord_pred)
                if (np.abs(mean_label-mean_prediction) < size/5).all():
                    iou = self.__compute_iou_from_contours(coord_pred,
                                                           coord_label)
                    if iou > best_iou:
                        best_iou = iou
                        best_prediction = index_pred
            if best_prediction is not None:
                if best_iou > ious[best_prediction]:
                    ious[best_prediction] = best_iou
        return ious

    def __mapping_objects(self) -> dict:
        """
        Map the predicted objects with the ground truth for each class.
        The objects with highest Intersection-over-Union value are mapped
        together.
        :return ious: The Intersection-over-Union values for the computed
                      mapping.
        """
        ious = {channel: None for channel in self.classes[1:]}
        for index, channel in enumerate(self.classes[1:], 1):
            coords_prediction = get_coords(self.prediction, index)
            coords_label = get_coords(self.label, index)
            self.positive_examples[channel] = len(coords_label)

            ious[channel] = self.__get_ious_by_channel(coords_prediction,
                                                       coords_label)
        return ious

    def __rank_predicted_objects(self):
        """
        Rank the predited objects by decreasing confidence score and decreasing
        Intersection-over-Union values.
        :return scores: The ranked objects confidence and
                        Intersection-over-Union.
        """
        ious = self.__mapping_objects()
        scores = {channel: {} for channel in self.classes[1:]}
        for index, channel in enumerate(self.classes[1:], 1):
            coords_prediction = get_coords(self.prediction, index)
            for index_coord, coord in enumerate(coords_prediction):
                conf = compute_confidence(coord,
                                          self.probabilities[index, :, :])
                scores[channel][index_coord] = conf

            tuples_score_iou = [(v, ious[channel][k])
                                for k, v in scores[channel].items()]
            scores[channel] = sorted(tuples_score_iou,
                                     key=lambda item: (-item[0], -item[1]))
        return scores

    def compute_rank_scores(self) -> dict:
        """
        Compute the number of true positive objects and the total of the
        predicted objects. It is used later to compute the overall precision,
        recall and F-score.
        :return scores: The scores obtained for a given rank, IoU
                        threshold and class.
        """
        ranked_objects = self.__rank_predicted_objects()
        scores = {
            channel: {
                iou: None for iou in range(50, 100, 5)
            } for channel in self.classes[1:]
        }
        for channel in self.classes[1:]:
            channel_scores = ranked_objects[channel]
            for iou in range(50, 100, 5):
                rank_scores = {
                    rank: {'True': 0, 'Total': 0} for rank in range(95, 45, -5)
                }
                for rank in range(95, 45, -5):
                    rank_objects = list(filter(
                        lambda item: item[0] >= rank/100, channel_scores))
                    rank_scores[rank]['True'] = sum(x[1] > iou / 100
                                                    for x in rank_objects)
                    rank_scores[rank]['Total'] = len(rank_objects)
                scores[channel][iou] = rank_scores
        return scores

    def update_rank_scores(self, scores: dict, positives: dict):
        """
        Update the global results by adding the current scores.
        :param scores: The dictionary containing the global scores.
        :param positives: The total number of ground truth objects by class.
        :return scores: The updated dictionary.
        :return positives: The updated number of objects.
        """
        for channel in self.classes[1:]:
            positives[channel] += self.positive_examples[channel]
            for iou in range(50, 100, 5):
                for rank in range(95, 45, -5):
                    scores[channel][iou][rank]['True'] += \
                        self.rank_scores[channel][iou][rank]['True']
                    scores[channel][iou][rank]['Total'] += \
                        self.rank_scores[channel][iou][rank]['Total']
        return scores, positives

    def __init_results(self) -> dict:
        """
        Initialize the results dictionnary by generating dictionary for
        the different rank and Intersection-over-Union thresholds.
        :return: The Initialized results dictionnary.
        """
        return {
            iou: {
                rank: 0 for rank in range(95, 45, -5)
            } for iou in range(50, 100, 5)
        }

    def get_average_precision(self, precisions: list, recalls: list) -> float:
        """
        Compute the mean average precision. Interpolate the precision-recall
        curve, then get the interpolated precisions for values [0, 0.1, 1].
        Compute the average precision over the range of values.
        :param precisions: The computed precisions for a given channel and a
                           given confidence score.
        :param recalls: The computed recalls for a given channel and a given
                        confidence score.
        :return mAP: The precision for the channel and for the confidence
                     score range averaged over 11 recalls values.
        """
        rp_tuples = []
        # Interpolated precision-recall curve.
        while len(precisions) > 0:
            max_precision = np.max(precisions)
            argmax_precision = np.argmax(precisions)
            max_recall = recalls[argmax_precision]
            rp_tuples.append({'p': max_precision, 'r': max_recall})
            for _ in range(argmax_precision+1):
                precisions.pop(0)
                recalls.pop(0)
        rp_tuples[-1]['r'] = 1

        rank_ap = {rank: None for rank in range(0, 11)}
        # Get precisions for the range.
        for rank in range(0, 11, 1):
            for rp_tuple in rp_tuples:
                if rank/10 <= rp_tuple['r'] and rank_ap[rank] is None:
                    rank_ap[rank] = rp_tuple['p']

        rank_ap = np.mean(list(rank_ap.values()))
        return rank_ap

    def get_mean_results(self, scores: dict, positives: dict,
                         results: dict) -> dict:
        """
        Get the mean metrics values for all the test set.
        :param scores: The overall computed scores.
        :param positives: The total number of ground truth objects by class.
        :param results: The dictionary containing all the computed results.
        :return results: The dictionary containing the mean computed values.
        """
        for channel in self.classes[1:]:
            current_scores = scores[channel]
            precisions = self.__init_results()
            recalls = self.__init_results()
            fscores = self.__init_results()
            aps = {iou: 0 for iou in range(50, 100, 5)}
            for iou in range(50, 100, 5):
                for rank in range(95, 45, -5):
                    if current_scores[iou][rank]['Total'] != 0:
                        precisions[iou][rank] = \
                            current_scores[iou][rank]['True'] / \
                            current_scores[iou][rank]['Total']
                    if positives[channel] != 0:
                        recalls[iou][rank] = current_scores[iou][rank]['True'] / \
                            positives[channel]
                    if precisions[iou][rank] + recalls[iou][rank] != 0:
                        fscores[iou][rank] = 2 * \
                            (precisions[iou][rank] * recalls[iou][rank]) / \
                            (precisions[iou][rank] + recalls[iou][rank])
                aps[iou] = self.get_average_precision(
                    list(precisions[iou].values()),
                    list(recalls[iou].values()))
            results['precision'][channel] = precisions
            results['recall'][channel] = recalls
            results['fscore'][channel] = fscores
            results['AP'][channel] = aps
        return results
