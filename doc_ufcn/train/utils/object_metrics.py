"""
The object metrics module
======================

Use it to compute different metrics during evaluation.
Available metrics:
    - Precision
    - Recall
    - F-score
    - Average precision
"""

import numpy as np


def __compute_iou_from_contours(poly1, poly2) -> float:
    """
    Get the Intersection-over-Union value between two contours.
    :param poly1: The first contour used to compute the
                  Intersection-over-Union.
    :param poly2: The second contour used to compute the
                  Intersection-over-Union.
    :return: The computed Intersection-over-Union value between the two
             contours.
    """
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    return intersection / union if union != 0 else 1


def __get_ious(labels: list, predictions: list) -> dict:
    """
    Get the Intersection-over-Union values of the predicted objects.
    :param labels: The coordinates of the ground truth objects.
    :param predictions: The coordinates of the predicted objects.
    :return ious: A dictionary with the highest Intersection-over-Union
                  value for each predicted object.
    """
    ious = {key: 0 for key in range(len(predictions))}

    for _, label in labels:
        best_iou = 0
        best_prediction = None
        for index, prediction in enumerate(predictions):
            iou = __compute_iou_from_contours(prediction[1], label)
            if iou > best_iou:
                best_iou = iou
                best_prediction = index
        if best_prediction is not None and best_iou > ious[best_prediction]:
            ious[best_prediction] = best_iou
    return ious


def __rank_predicted_objects(labels: list, predictions: list) -> dict:
    """
    Rank the predited objects by decreasing confidence score and decreasing
    Intersection-over-Union values.
    :param labels: The coordinates and confidence scores of the ground truth objects.
    :param predictions: The coordinates and confidence scores of the predicted objects.
    :return scores: The ranked predicted objects by confidence and
                    Intersection-over-Union.
    """
    ious = __get_ious(labels, predictions)

    scores = {index: prediction[0] for index, prediction in enumerate(predictions)}
    tuples_score_iou = [(v, ious[k]) for k, v in scores.items()]
    scores = sorted(tuples_score_iou, key=lambda item: (-item[0], -item[1]))
    return scores


def compute_rank_scores(labels: list, predictions: list, classes: list) -> dict:
    """
    Compute the number of true positive objects and the total of the
    predicted objects. It is used later to compute the overall precision,
    recall and F-score.
    :param labels: The coordinates and confidence scores of the ground truth objects.
    :param predictions: The coordinates and confidence scores of the predicted objects.
    :param classes: The classes names involved in the experiment.
    :return scores: The scores obtained for a each rank, IoU
                    threshold and class.
    """
    scores = {channel: {iou: None for iou in range(50, 100, 5)} for channel in classes}
    for channel in classes:
        channel_scores = __rank_predicted_objects(labels[channel], predictions[channel])
        for iou in range(50, 100, 5):
            rank_scores = {rank: {"True": 0, "Total": 0} for rank in range(95, -5, -5)}
            for rank in range(95, -5, -5):
                rank_objects = list(
                    filter(lambda item: item[0] >= rank / 100, channel_scores)
                )
                rank_scores[rank]["True"] = sum(x[1] > iou / 100 for x in rank_objects)
                rank_scores[rank]["Total"] = len(rank_objects)
            scores[channel][iou] = rank_scores
    return scores


def update_rank_scores(global_scores: dict, image_scores: dict, classes: list) -> dict:
    """
    Update the global scores by adding page scores.
    :param global_scores: The scores obtained so far.
    :param image_scores: the current page scores.
    :param classes: The classes names involved in the experiment.
    :return global_scores: The updated global scores.
    """
    for channel in classes:
        for iou in range(50, 100, 5):
            for rank in range(95, -5, -5):
                global_scores[channel][iou][rank]["True"] += image_scores[channel][iou][
                    rank
                ]["True"]
                global_scores[channel][iou][rank]["Total"] += image_scores[channel][
                    iou
                ][rank]["Total"]
    return global_scores


def __init_results() -> dict:
    """
    Initialize the results dictionary by generating dictionary for
    the different rank and Intersection-over-Union thresholds.
    :return: The initialized results dictionary.
    """
    return {iou: {rank: 0 for rank in range(95, -5, -5)} for iou in range(50, 100, 5)}


def __get_average_precision(precisions: list, recalls: list) -> float:
    """
    Compute the mean average precision. Interpolate the precision-recall
    curve, then get the interpolated precisions for values.
    Compute the average precision.
    :param precisions: The computed precisions for a given channel and a
                       given confidence score.
    :param recalls: The computed recalls for a given channel and a given
                    confidence score.
    :return: The average precision for the channel and for the confidence
             score range.
    """
    rp_tuples = []
    # Interpolated precision-recall curve.
    while len(precisions) > 0:
        max_precision = np.max(precisions)
        argmax_precision = np.argmax(precisions)
        max_recall = recalls[argmax_precision]
        rp_tuples.append({"p": max_precision, "r": max_recall})
        for _ in range(argmax_precision + 1):
            precisions.pop(0)
            recalls.pop(0)
    rp_tuples[-1]["r"] = 1

    ps = [rp_tuple["p"] for rp_tuple in rp_tuples]
    rs = [rp_tuple["r"] for rp_tuple in rp_tuples]
    ps.insert(0, ps[0])
    rs.insert(0, 0)
    return np.trapz(ps, x=rs)


def get_mean_results(
    global_scores: dict, true_gt: dict, classes: list, results: dict
) -> dict:
    """
    Get the mean metrics values for all the set.
    :param global_scores: The overall computed scores.
    :param true_gt: The total number of ground truth objects by class.
    :param classes: The classes names involved in the experiment.
    :param results: The initialized results dictionary.
    :return results: The dictionary containing the mean computed values.
    """
    for channel in classes:
        precisions = __init_results()
        recalls = __init_results()
        fscores = __init_results()
        aps = {iou: 0 for iou in range(50, 100, 5)}
        for iou in range(50, 100, 5):
            for rank in range(95, -5, -5):
                true_predicted = global_scores[channel][iou][rank]["True"]
                predicted = global_scores[channel][iou][rank]["Total"]

                precisions[iou][rank] = (
                    true_predicted / predicted if predicted != 0 else 1
                )
                recalls[iou][rank] = (
                    true_predicted / true_gt[channel] if true_gt[channel] != 0 else 1
                )

                if precisions[iou][rank] + recalls[iou][rank] != 0:
                    fscores[iou][rank] = (
                        2
                        * (precisions[iou][rank] * recalls[iou][rank])
                        / (precisions[iou][rank] + recalls[iou][rank])
                    )
            aps[iou] = __get_average_precision(
                list(precisions[iou].values()), list(recalls[iou].values())
            )
            results[channel]["precision"] = precisions
            results[channel]["recall"] = recalls
            results[channel]["fscore"] = fscores
            results[channel]["AP"] = aps
    return results
