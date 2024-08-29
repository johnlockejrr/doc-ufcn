"""
The evaluation module
======================

Use it to evaluation a trained network.
"""

import logging
import time
from pathlib import Path

import mlflow
import numpy as np
from tqdm import tqdm

import doc_ufcn.train.utils.evaluation as ev_utils
import doc_ufcn.train.utils.object_metrics as o_metrics
import doc_ufcn.train.utils.pixel_metrics as p_metrics
from doc_ufcn.utils import read_json


def run(
    log_path: Path,
    classes_names: list,
    set: str,
    data_paths: dict,
    dataset: str,
    prediction_path: Path,
    evaluation_path: Path,
    mlflow_logging: bool,
):
    """
    Run the evaluation.
    :param log_path: Path to save the evaluation results and load the model.
    :param classes_names: The names of the classes involved during the experiment.
    :param set: The current evaluated set.
    :param data_paths: Path to the data folders.
    :param dataset: The dataset to evaluate.
    :param prediction_path: Path where the prediction has been written.
    :param evaluation_path: Path where the evaluation will been written.
    :param mlflow_logging: Whether we should log data to MLflow.
    """
    # Run evaluation.
    logging.info(f"Starting evaluation: {dataset}")
    starting_time = time.time()

    label_dir = [dir for dir in data_paths if dataset in str(dir)][0]

    pixel_metrics = {
        channel: {metric: [] for metric in ["iou", "precision", "recall", "fscore"]}
        for channel in classes_names[1:]
    }
    object_metrics = {
        channel: {metric: {} for metric in ["precision", "recall", "fscore", "AP"]}
        for channel in classes_names[1:]
    }
    rank_scores = {
        channel: {
            iou: {rank: {"True": 0, "Total": 0} for rank in range(95, -5, -5)}
            for iou in range(50, 100, 5)
        }
        for channel in classes_names[1:]
    }
    number_of_gt = {channel: 0 for channel in classes_names[1:]}
    for img_path in tqdm(label_dir.iterdir(), desc=f"Evaluation (prog) {set}"):
        gt_regions = read_json(img_path)
        pred_regions = read_json(
            log_path / prediction_path / set / dataset / img_path.name
        )

        gt_polys = ev_utils.get_polygons(gt_regions, classes_names)

        if gt_regions["img_size"] != pred_regions["img_size"]:
            pred_regions = ev_utils.resize_polygons(
                pred_regions, gt_regions["img_size"], pred_regions["img_size"]
            )
        pred_polys = ev_utils.get_polygons(pred_regions, classes_names)

        pixel_metrics = p_metrics.compute_metrics(
            gt_polys, pred_polys, classes_names[1:], pixel_metrics
        )

        image_rank_scores = o_metrics.compute_rank_scores(
            gt_polys, pred_polys, classes_names[1:]
        )
        rank_scores = o_metrics.update_rank_scores(
            rank_scores, image_rank_scores, classes_names[1:]
        )
        number_of_gt = {
            channel: number_of_gt[channel] + len(gt_polys[channel])
            for channel in classes_names[1:]
        }

    object_metrics = o_metrics.get_mean_results(
        rank_scores, number_of_gt, classes_names[1:], object_metrics
    )

    if mlflow_logging:
        # Log metrics per channel
        for channel in classes_names[1:]:
            prefix = f"{set.upper()} {channel}"
            # Pixel metrics
            prefixed_pixel_metrics = {
                f"{prefix}_{metric}": np.round(np.mean(value), 4)
                for metric, value in pixel_metrics[channel].items()
            }
            aps = object_metrics[channel]["AP"]
            # AP values
            AP_metrics = {
                f"{prefix} AP IOU_0.{level}": np.round(aps[level], 4)
                for level in [50, 75, 95]
            }
            AP_metrics[f"{prefix} AP_0.5-0.95"] = np.round(
                np.mean(list(aps.values())), 4
            )
            mlflow.set_tags(tags={**prefixed_pixel_metrics, **AP_metrics})

    # Print the results.
    print(set)
    for channel in classes_names[1:]:
        print(channel)
        print("IoU       = ", np.round(np.mean(pixel_metrics[channel]["iou"]), 4))
        print("Precision = ", np.round(np.mean(pixel_metrics[channel]["precision"]), 4))
        print("Recall    = ", np.round(np.mean(pixel_metrics[channel]["recall"]), 4))
        print("F-score   = ", np.round(np.mean(pixel_metrics[channel]["fscore"]), 4))

        aps = object_metrics[channel]["AP"]
        print("AP [IOU=0.50] = ", np.round(aps[50], 4))
        print("AP [IOU=0.75] = ", np.round(aps[75], 4))
        print("AP [IOU=0.95] = ", np.round(aps[95], 4))
        print("AP [0.5,0.95] = ", np.round(np.mean(list(aps.values())), 4))
        print("\n")

    (log_path / evaluation_path / set).mkdir(exist_ok=True, parents=True)
    # ev_utils.save_graphical_results(
    #     object_metrics,
    #     classes_names[1:],
    #     log_path / params.evaluation_path / set
    # )
    ev_utils.save_results(
        pixel_metrics,
        object_metrics,
        classes_names[1:],
        log_path / evaluation_path / set,
        dataset,
    )

    end = time.gmtime(time.time() - starting_time)
    logging.info(
        "Finished evaluating in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
    )
