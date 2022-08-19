# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pytest

from doc_ufcn.train.mask import generate_mask


@pytest.mark.parametrize(
    "image_width, image_height, label_polygons, label_colors, output_path, ground_truth_path",
    [
        (
            400,
            500,
            {
                "text_line": [
                    [[41, 280], [361, 280], [361, 311], [41, 311], [41, 280]]
                ],
                "picture": [[[58, 76], [307, 76], [307, 205], [58, 205], [58, 76]]],
            },
            {"text_line": (255, 0, 0), "picture": (0, 0, 255)},
            "doc_ufcn/train/test.png",
            "doc_ufcn/train/ground_truth.png",
        )
    ],
)
def test_generate_mask(
    image_width,
    image_height,
    label_polygons,
    label_colors,
    output_path,
    ground_truth_path,
):
    generate_mask(image_width, image_height, label_polygons, label_colors, output_path)
    image_path = output_path[:-4] + "_mask.png"
    print(ground_truth_path)
    truth_image = cv2.imread(ground_truth_path)
    image = cv2.imread(image_path)

    print("image : ", image.shape)
    print("truth_image : ", truth_image.shape)
    errorL2 = cv2.norm(truth_image, image, cv2.NORM_L2)
    similarity = 1 - errorL2 / (image_height * image_width)
    print("Similarity = ", similarity)

    print("truth_image : ", np.unique(truth_image))
    print("image : ", np.unique(image))
    assert similarity == 1
