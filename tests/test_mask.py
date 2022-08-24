# -*- coding: utf-8 -*-
import tempfile

import cv2
import pytest
from skimage.metrics import structural_similarity as ssim

from doc_ufcn.train.mask import generate_mask


@pytest.mark.parametrize(
    "image_width, image_height, label_polygons, label_colors",
    [
        (
            400,
            500,
            {
                "text_line": [
                    [[41, 280], [361, 280], [361, 311], [41, 311], [41, 280]]
                ],
                "paragraph": [[[58, 76], [307, 76], [307, 205], [58, 205], [58, 76]]],
            },
            {"text_line": (255, 0, 0), "paragraph": (0, 0, 255)},
        )
    ],
)
def test_generate_mask(
    image_width, image_height, label_polygons, label_colors, expected_mask_path
):
    # Read the expected output image
    expected_mask = cv2.imread(expected_mask_path)

    # Generate the mask
    _, output_path = tempfile.mkstemp(suffix=".teklia.test.mask.png")
    generate_mask(
        image_width, image_height, None, label_polygons, label_colors, output_path
    )
    generated_mask = cv2.imread(output_path)

    assert generated_mask.shape == expected_mask.shape

    # Image comparison is done with skimage.structural_similarity
    similarity = ssim(
        expected_mask,
        generated_mask,
        data_range=generated_mask.max() - generated_mask.min(),
        channel_axis=2,
    )
    assert similarity >= 0.95
