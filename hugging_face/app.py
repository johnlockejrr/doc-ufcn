# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from doc_ufcn import models
from doc_ufcn.main import DocUFCN
from hugging_face.config import parse_configurations

parser = argparse.ArgumentParser(description="UFCN HuggingFace app")
parser.add_argument(
    "--config",
    type=Path,
    required=True,
    help="The JSON-formatted configuration file of the Hugging Face app",
)

# Get the application's public mode (local or with sharing)
parser.add_argument(
    "--public",
    action="store_true",
    help="Generate a shareable link to your application",
)

# Parse arguments
args = parser.parse_args()

# Load the config
config = parse_configurations(args.config)

# Download the model
model_path, parameters = models.download_model(name=config["model_name"])

# Store classes_colors list
classes_colors = config["classes_colors"]

# Store classes
classes = parameters["classes"]

# Check that the number of colors is equal to the number of classes -1
assert len(classes) - 1 == len(
    classes_colors
), f"The parameter classes_colors was filled with the wrong number of colors. {len(classes)-1} colors are expected instead of {len(classes_colors)}."

# Check that the paths of the examples are valid
for example in config["examples"]:
    assert os.path.exists(example), f"The path of the image '{example}' does not exist."

# Load the model
model = DocUFCN(
    no_of_classes=len(classes),
    model_input_size=parameters["input_size"],
    device="cpu",
)
model.load(model_path=model_path, mean=parameters["mean"], std=parameters["std"])


def query_image(image):
    """
    Draws the predicted polygons with the color provided by the model on an image

    :param image: An image to predict
    :return: Image, an image with the predictions
    """

    # Make a prediction with the model
    detected_polygons, probabilities, mask, overlap = model.predict(
        input_image=image, raw_output=True, mask_output=True, overlap_output=True
    )

    # Load image
    image = Image.fromarray(image)

    # Make a copy of the image to keep the source and also to be able to use Pillow's blend method
    img2 = image.copy()

    # Create the polygons on the copy of the image for each class with the corresponding color
    # We do not draw polygons of the background channel (channel 0)
    for channel in range(1, len(classes)):
        for polygon in detected_polygons[channel]:
            # Draw the polygons on the image copy.
            # Loop through the class_colors list (channel 1 has color 0)
            ImageDraw.Draw(img2).polygon(
                polygon["polygon"], fill=classes_colors[channel - 1]
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

# Launch the application with the public mode (True or False)
process_image.launch(share=args.public)
