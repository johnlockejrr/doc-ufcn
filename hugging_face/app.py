# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from doc_ufcn import models
from doc_ufcn.main import DocUFCN
from hugging_face.config import parse_configurations

# Get the configuration json file with a cli
parser = argparse.ArgumentParser(description="UFCN HuggingFace app")
parser.add_argument(
    "--config", type=Path, required=True, help="A .json config for Hugging Face app"
)

# Get the application's link sharing mode (simply local or with sharing)
parser.add_argument(
    "--share-link",
    action="store_true",
    help="Boolean representing the activation of the link for share with other users ",
)

# Parse arguments
args = parser.parse_args()

# Load the config
config = parse_configurations(args.config)

# Download the model
model_path, parameters = models.download_model(config["model_name"])

# Check that the number of colors is equal to the number of classes
assert len(parameters["classes"]) == len(
    config["classes_colors"]
), f"The parameter prediction_color was filled with the wrong number of colors. {len(parameters['classes'])} colors are expected instead of {len(config['classes_colors'])}"

# Load the model
model = DocUFCN(len(parameters["classes"]), parameters["input_size"], "cpu")
model.load(model_path, parameters["mean"], parameters["std"])


def query_image(image):
    """
    Get an input image and process it with predictions

    :param image: An image to predict
    :return: Image, an image with the predictions
    """

    # Make a prediction with the model
    detected_polygons, probabilities, mask, overlap = model.predict(
        image, raw_output=True, mask_output=True, overlap_output=True
    )

    # Load image
    image = Image.fromarray(image)

    # Copy image
    img2 = image.copy()

    # Create the polygons on the copy of the image for each class with the corresponding color
    for channel in range(1, len(parameters["classes"])):
        for polygon in detected_polygons[channel]:
            ImageDraw.Draw(img2).polygon(
                polygon["polygon"], fill=config["classes_colors"][channel - 1]
            )

    # Return the blend of the images
    return Image.blend(image, img2, 0.5)


# Create an interface with the config
process_image = gr.Interface(
    fn=query_image,
    inputs=[gr.Image()],
    outputs=[gr.Image()],
    title=config["title"],
    description=config["description"],
    examples=config["examples"],
)

# Launch the application with the shared link option retrieved in the config (True or False)
process_image.launch(share=args.share_link)
