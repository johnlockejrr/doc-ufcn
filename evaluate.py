#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The evaluation module
    ======================

    Use it to evaluation a trained network.
"""

import os
import logging
import time
import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
import torch
import utils.evaluation_utils as ev_utils
import utils.pixel_metrics as p_metrics
import utils.object_metrics as o_metrics


def run(log_path: str, classes_names: list, params: dict):
    """
    Run the evaluation.
    :param log_path: Path to save the evaluation results and load the model.
    :param classes_names: The names of the classes involved during the experiment.
    :param params: The evaluation parameters.
    """
    # Run evaluation.
    logging.info('Starting evaluation')
    starting_time = time.time()
    
    for set, label in zip(['train', 'val', 'test'],
                          [params.train_gt_path, params.val_gt_path, params.test_gt_path]):
        pixel_metrics = {channel: {metric: [] for metric in ['iou', 'precision', 'recall', 'fscore']} for channel in classes_names[1:]}
        object_metrics = {channel: {metric: {} for metric in ['precision', 'recall', 'fscore', 'AP']} for channel in classes_names[1:]}
        rank_scores = {
            channel: {
                iou: {
                    rank: {'True': 0, 'Total': 0} for rank in range(100, -5, -5)
                } for iou in range(50, 100, 5)
            } for channel in classes_names[1:]}
        number_of_gt = {channel: 0 for channel in classes_names[1:]}
        for img_name in tqdm(os.listdir(label), desc="Evaluation (prog) "+set):
            gt_regions = ev_utils.read_json(os.path.join(label, img_name))
            pred_regions = ev_utils.read_json(os.path.join(log_path, params.prediction_path, set, img_name))
            assert(gt_regions['img_size'] == pred_regions['img_size'])
            gt_polys = ev_utils.get_polygons(gt_regions, classes_names)
            pred_polys = ev_utils.get_polygons(pred_regions, classes_names)

            pixel_metrics = p_metrics.compute_metrics(gt_polys, pred_polys, classes_names[1:], pixel_metrics)

            image_rank_scores = o_metrics.compute_rank_scores(gt_polys, pred_polys, classes_names[1:])
            rank_scores = o_metrics.update_rank_scores(rank_scores, image_rank_scores, classes_names[1:])
            number_of_gt = {channel: number_of_gt[channel]+len(gt_polys[channel]) for channel in classes_names[1:]}
        
        object_metrics = o_metrics.get_mean_results(rank_scores, number_of_gt, classes_names[1:], object_metrics)

        # Print the results.
        print(set)
        for channel in classes_names[1:]:
            print(channel)
            print('IoU       = ', np.round(np.mean(pixel_metrics[channel]['iou']), 4))
            print('Precision = ', np.round(np.mean(pixel_metrics[channel]['precision']), 4))
            print('Recall    = ', np.round(np.mean(pixel_metrics[channel]['recall']), 4))
            print('F-score   = ', np.round(np.mean(pixel_metrics[channel]['fscore']), 4))

            aps = object_metrics[channel]['AP']
            print('AP [IOU=0.50] = ', np.round(aps[50], 4))
            print('AP [IOU=0.75] = ', np.round(aps[75], 4))
            print('AP [IOU=0.95] = ', np.round(aps[95], 4))
            print('AP [0.5,0.95] = ', np.round(np.mean(list(aps.values())), 4))
            print('\n')

        os.makedirs(os.path.join(log_path, params.evaluation_path, set), exist_ok=True)
        #ev_utils.save_graphical_results(object_metrics, classes_names[1:],
        #                                os.path.join(log_path, params.evaluation_path, set))
        ev_utils.save_results(pixel_metrics, object_metrics, classes_names[1:],
                              os.path.join(log_path, params.evaluation_path, set))

    end = time.gmtime(time.time() - starting_time)
    logging.info('Finished predicting in %2d:%2d:%2d',
                 end.tm_hour, end.tm_min, end.tm_sec)

