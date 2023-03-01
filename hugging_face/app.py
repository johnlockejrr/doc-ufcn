# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
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
    :return: Image and dict, an image with the predictions and a
        dictionary mapping an object idx (starting from 1) to a dictionary describing the detected object:
        - `polygon` key : list, the coordinates of the points of the polygon,
        - `confidence` key : float, confidence of the model,
        - `channel` key : str, the name of the predicted class.
    """

    # Make a prediction with the model
    detected_polygons, probabilities, mask, overlap = model.predict(
        input_image=image, raw_output=True, mask_output=True, overlap_output=True
    )

    # Load image
    image = Image.fromarray(image)

    # Make a copy of the image to keep the source and also to be able to use Pillow's blend method
    img2 = image.copy()

    # Initialize the dictionary which will display the json on the application
    predict = []

    # Create the polygons on the copy of the image for each class with the corresponding color
    # We do not draw polygons of the background channel (channel 0)
    for channel in range(1, len(classes)):
        for i, polygon in enumerate(detected_polygons[channel]):
            # Draw the polygons on the image copy.
            # Loop through the class_colors list (channel 1 has color 0)
            ImageDraw.Draw(img2).polygon(
                polygon["polygon"], fill=classes_colors[channel - 1]
            )

            # Build the dictionary
            # Add an index to dictionary keys to differentiate predictions of the same class
            predict.append(
                {
                    # The list of coordinates of the points of the polygon.
                    # Use np.asarray to change list to ndarray to transform int64 to int32 with
                    # np.ndarray.tolist to be able to serialize it to JSON.
                    "polygon": np.asarray(polygon["polygon"], dtype=np.int32).tolist(),
                    # Confidence that the model predicts the polygon in the right place
                    "confidence": polygon["confidence"],
                    # The channel on which the polygon is predicted
                    "channel": classes[channel],
                }
            )

    # Return the blend of the images and the dictionary formatted in json
    return Image.blend(image, img2, 0.5), json.dumps(predict, indent=20)


with gr.Blocks() as process_image:

    # Create app title
    gr.Markdown(f"# {config['title']}")

    # Create app description
    gr.Markdown(config["description"])

    # Create a first row of blocks
    with gr.Row():

        # Create a column on the left
        with gr.Column():

            # Generates an image that can be uploaded by a user
            image = gr.Image()

            # Create a row under the image
            with gr.Row():

                # Generate a button to clear the inputs and outputs
                clear_button = gr.Button("Clear", variant="secondary")

                # Generates a button to submit the prediction
                submit_button = gr.Button("Submit", variant="primary")

            # Create a row under the buttons
            with gr.Row():

                # Generate example images that can be used as input image
                examples = gr.Examples(inputs=image, examples=config["examples"])

        # Create a column on the right
        with gr.Column():

            # Generates an output image that does not support upload
            image_output = gr.Image(interactive=False)

            # Create a row under the predicted image
            with gr.Row():

                # Create a column so that the JSON output doesn't take the full size of the page
                with gr.Column():

                    # Create a collapsible region
                    with gr.Accordion("JSON"):

                        # Generates a json with the model predictions
                        json_output = gr.JSON()

    # Create the button to clear the inputs and outputs
    clear_button.click(
        lambda x, y, z: (None, None, None),
        inputs=[image, image_output, json_output],
        outputs=[image, image_output, json_output],
    )

    # Create the button to submit the prediction
    submit_button.click(query_image, inputs=image, outputs=[image_output, json_output])

# Launch the application with the public mode (True or False)
process_image.launch(share=args.public)
