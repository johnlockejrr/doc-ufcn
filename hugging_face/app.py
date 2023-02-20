# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from doc_ufcn import models
from doc_ufcn.main import DocUFCN
from hugging_face.config import get_config

parser = argparse.ArgumentParser(description="UFCN HuggingFace app")
parser.add_argument(
    "--config", type=Path, required=True, help="A .json config for Hugging Face app"
)

args = parser.parse_args()

# Load model with config
config = get_config(args.config)
model_path, parameters = models.download_model(config["model_name"])
model = DocUFCN(len(parameters["classes"]), parameters["input_size"], "cpu")
model.load(model_path, parameters["mean"], parameters["std"])


def query_image(image):
    """
    Get an input image and process it with predictions

    :param image: An image to predict
    :return: Image, an image with the predictions
    """
    detected_polygons, probabilities, mask, overlap = model.predict(
        image, raw_output=True, mask_output=True, overlap_output=True
    )

    image = Image.fromarray(image)
    img2 = image.copy()

    for channel in range(1, config["no_of_classes"]):
        print(channel)
        for polygon in detected_polygons[channel]:
            ImageDraw.Draw(img2).polygon(
                polygon["polygon"], fill=config["prediction_color"][channel - 1]
            )

    return Image.blend(image, img2, 0.5)


process_image = gr.Interface(
    fn=query_image,
    inputs=[gr.Image()],
    outputs=[gr.Image()],
    title=config["title"],
    description=config["description"],
    examples=config["examples"],
)

process_image.launch(share=True)
