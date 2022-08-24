# -*- coding: utf-8 -*-
from itertools import combinations
from pathlib import Path

import cv2
import numpy
from shapely.affinity import scale
from shapely.geometry import LineString, Polygon
from shapely.geometry.multipolygon import MultiPolygon


def new_image_size(input_size: tuple, output_size: int):
    """
    Get the resize label size and corresponding ratios.
    The longest side of the image will have a length of output_size.
    :param input_size: The original image size.
    :param output_size: The output maximum size.
    """
    # Compute the new image size.
    ratio = float(output_size) / max(input_size[:2])
    new_size = tuple([int(x * ratio) for x in input_size[:2]])
    # Compute the ratios between the original and new image sizes.
    # It is used to resize the annotations.
    ratios = [new_size[index] / input_size[index] for index in range(len(new_size))]
    return *new_size, *ratios


def generate_mask(
    image_width: int,
    image_height: int,
    max_image_size: float,
    label_polygons: dict,
    label_colors: dict,
    output_path: Path,
) -> str:
    """
    Generate a mask with the given dimensions and polygons.
    Returns the path to the generated image.
    """
    if max_image_size is not None:
        mask_height, mask_width, ratio_height, ratio_width = new_image_size(
            [image_height, image_width],
            max_image_size,
        )
    else:
        mask_height, mask_width, ratio_height, ratio_width = (
            image_height,
            image_width,
            1,
            1,
        )
    # Remove the extension from the output_path and add the suffix
    black_img = numpy.zeros(
        (mask_height, mask_width, 3),
        dtype=numpy.uint8,
    )
    cv2.imwrite(output_path, black_img)
    img = cv2.imread(output_path)

    # Draw the polygons on the image
    for label in label_polygons:
        # Retrieve polygon color
        color = label_colors[label]
        # Retrieve corresponding polygons
        polygons = [Polygon(poly) for poly in label_polygons[label]]

        # Resize the polygons
        polygons = resize_polygons(
            polygons=polygons, height=ratio_height, width=ratio_width
        )

        # Split the polygons
        polygons = split_polygons(polygons)

        # Draw the polygons on the mask image
        draw_polygons(polygons, img, color)

    # Update the mask image on Disk
    cv2.imwrite(output_path, img)


def draw_polygons(polygons: list, img, color: tuple) -> None:
    """
    Draw the provided polygons on the image
    """
    for poly in polygons:
        if isinstance(poly, MultiPolygon):
            draw_polygons(polygons=list(poly.geoms), img=img, color=color)
        else:
            contours = [numpy.array(poly.exterior.coords).round().astype(numpy.int32)]
            for contour in contours:
                if len(contour) > 0:
                    cv2.drawContours(img, contours, 0, tuple(reversed(color)), -1)


def resize_polygons(polygons: list, height: float, width: float) -> list:
    """
    Resize the polygons.
    :param polygons: The polygons to resize.
    :param height: The ratio in the height dimension.
    :param width: The ratio in the width dimension.
    :return: A list of the resized polygons.
    """
    return [
        scale(polygon, xfact=width, yfact=height, origin=(0, 0)) for polygon in polygons
    ]


def split_polygons(polygons: list) -> list:
    """
    Split the touching and overlapping polygons.
    :param polygons: The polygons to split.
    :return polygons: The non-touching polygons.
    """
    eps = 2
    for comb in combinations(range(len(polygons)), 2):
        poly1 = polygons[comb[0]]
        poly2 = polygons[comb[1]]
        # Skip invalid polygons as they cannot be compared.
        if not poly1.is_valid or not poly2.is_valid:
            continue
        # If the two polygons intersect: first erode them, then check if they still intersect.
        if poly1.intersects(poly2):
            poly1 = poly1.buffer(-eps)
            poly2 = poly2.buffer(-eps)
            intersection = poly1.intersection(poly2)
            # If they still intersect, remove the intersection from the biggest polygon.
            if not intersection.is_empty:
                if (
                    isinstance(intersection, Polygon)
                    and intersection.area < 0.2 * poly1.area
                    and intersection.area < 0.2 * poly2.area
                    or isinstance(intersection, MultiPolygon)
                ):
                    if poly1.area > poly2.area:
                        polygons[comb[0]] = poly1.difference(intersection)
                        polygons[comb[1]] = poly2.buffer(-eps)
                    else:
                        polygons[comb[1]] = poly2.difference(intersection)
                        polygons[comb[0]] = poly1.buffer(-eps)
                elif isinstance(intersection, LineString):
                    polygons[comb[0]] = poly1.difference(intersection)
                    polygons[comb[1]] = poly2.difference(intersection)
        elif poly1.touches(poly2):
            polygons[comb[0]] = poly1.buffer(-2 * eps)
            polygons[comb[1]] = poly2.buffer(-2 * eps)
    # Erode all polygons so that they don't touch when drawn over the label image.
    polygons = [poly.buffer(-2 * eps) for poly in polygons]
    return polygons
