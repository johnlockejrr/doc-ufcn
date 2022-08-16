# -*- coding: utf-8 -*-
import pytest
from doc_ufcn.train.mask import generate_mask
import cv2

@pytest.mark.parametrize(
    "image_width, image_height, label_polygons, label_colors, output_path, true_image_path",
    [
        (
            2600, 
            2102,
            {
                "text_line": [
                    [[0,0], [100, 0], [100, 100], [0,100], [0,0]]
                ],
                "picture": [
                    [[100, 100], [500, 100], [500, 500], [100,500], [100,100]]
                ],
            },
            {
                "text_line": (255, 0, 0), # red
                "picture": (0, 0, 255), # blue
            },
            "doc_ufcn/train/test.jpg"
        )
    ],
)
def test_generate_mask(image_width, image_height, label_polygons, label_colors, output_path, true_image_path):
    generate_mask(image_width, image_height, label_polygons, label_colors, output_path, true_image_path)
    image_path = output_path[:-4] + "_mask.jpg"
    true_image = cv2.imread(true_image_path)
    image = cv2.imread(image_path)
    errorL2 = cv2.norm(true_image, image, cv2.NORM_L2 )
    similarity = 1 - errorL2 / ( image_height * image_width )
    print('Similarity = ',similarity)
    assert similarity == 1
