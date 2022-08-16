# -*- coding: utf-8 -*-
from itertools import combinations
# from shapely.affinity import scale
from shapely.geometry import LineString, MultiPolygon, Polygon
import cv2
import numpy

def generate_mask(image_width, image_height, label_polygons, label_colors, output_path):

    image_path = str(output_path)[:-4] + "_mask.jpg"
    print(image_path)
    black_img = numpy.zeros(
        (image_height, image_width, 3),
        dtype=numpy.uint8,
    )

    line_polygons = label_polygons["text_line"]
    picture_polygons = label_polygons["picture"]
    line_color = label_colors["text_line"]
    picture_color = label_colors["picture"]

    # Prepare polygons for label image.
    line_polygons = [Polygon(poly) for poly in line_polygons]
    picture_polygons = [Polygon(poly) for poly in picture_polygons]
    # Resize the polygons
    # line_polygons = resize_polygons(polygons=line_polygons, height=1, width=1)
    # picture_polygons = resize_polygons(polygons=picture_polygons, height=1, width=1)
    # Split the polygons
    line_polygons = split_polygons(line_polygons)
    picture_polygons = split_polygons(picture_polygons)

    img = cv2.imwrite(image_path, black_img)
    img = cv2.imread(image_path)
    for line_poly in line_polygons:
      line_contours = [numpy.array(line_poly.exterior.coords).round().astype(numpy.int32)]
      for line_contour in line_contours:
          if len(line_contour) > 0:
            img = cv2.drawContours(img, line_contours, 0, tuple(reversed(line_color)), -1)
      cv2.imwrite(image_path, img)

    for pict_poly in picture_polygons:
      pict_contours = [numpy.array(pict_poly.exterior.coords).round().astype(numpy.int32)]
      for contour in pict_contours:
          if len(contour) > 0:
            img = cv2.drawContours(img, pict_contours, 0, tuple(reversed(picture_color)), -1)
      cv2.imwrite(image_path, img)

# def resize_polygons(polygons: list, height: float, width: float) -> list:
#     """
#     Resize the polygons.
#     :param polygons: The polygons to resize.
#     :param height: The ratio in the height dimension.
#     :param width: The ratio in the width dimension.
#     :return: A list of the resized polygons.
#     """
#     return [
#         scale(polygon, xfact=width, yfact=height, origin=(0, 0)) for polygon in polygons
#     ]

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


label_polygons = {
  "text_line": [
    [[0,0], [100, 0], [100, 100], [0,100], [0,0]]
  ],
  "picture": [
    [[100, 100], [500, 100], [500, 500], [100,500], [100,100]]
  ],
}
label_colors = {
  "text_line": (255, 0, 0), # red
  "picture": (0, 0, 255), # blue
}
generate_mask(2600, 2102, label_polygons, label_colors, "doc_ufcn/train/test_ufcn.jpg") 