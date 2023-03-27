# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from doc_ufcn import models
from doc_ufcn.main import DocUFCN
from hugging_face.config import parse_configurations

# Create an argument parser for get the config
parser = argparse.ArgumentParser(description="UFCN HuggingFace app")
parser.add_argument(
    "--config",
    type=Path,
    required=True,
    help="The YAML-formatted configuration file of the Hugging Face app",
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

# Create a list of models name
models_name = [model["model_name"] for model in config["models"]]


class UFCNModel(NamedTuple):
    name: str
    classes: list
    colors: list
    model: DocUFCN
    title: str
    description: str

    def get_class_name(self, channel_idx):
        return self.classes[channel_idx]


# Cached models, maps model_name to UFCNModel object
MODELS = {}


def load_model(model_name):
    """
    Load a model by name if it doesn't already exist then return the model

    :param model_name: The name of the selected model
    :return: The UFCNModel instance selected
    """

    if model_name in MODELS:
        # If cached already, return cached model
        return MODELS[model_name]

    # Create a new model

    # Download the model
    model_path, parameters = models.download_model(name=model_name)

    model_id = get_model_idx(model_name)

    # Store classes_colors list
    classes_colors = config["models"][model_id]["classes_colors"]

    # Store title
    title = config["models"][model_id]["title"]

    # Store description
    description = config["models"][model_id]["description"]

    # Store classes
    classes = parameters["classes"]

    # Check that the number of colors is equal to the number of classes -1
    assert len(classes) - 1 == len(
        classes_colors
    ), f"The parameter classes_colors was filled with the wrong number of colors. {len(classes)-1} colors are expected instead of {len(classes_colors)}."

    # Check that the paths of the examples are valid
    for example in config["examples"]:
        assert Path.exists(
            Path(example)
        ), f"The path of the image '{example}' does not exist."

    # Load the model
    model = DocUFCN(
        no_of_classes=len(classes),
        model_input_size=parameters["input_size"],
        device="cpu",
    )
    model.load(model_path=model_path, mean=parameters["mean"], std=parameters["std"])

    MODELS[model_name] = UFCNModel(
        name=model_name,
        classes=classes,
        colors=classes_colors,
        model=model,
        title=title,
        description=description,
    )
    return MODELS[model_name]


def query_image(model, image):
    """
    Load a model and draws the predicted polygons with the color provided by the model on an image

    :param model: A model selected in dropdown
    :param image: An image to predict
    :return: Image and dict, an image with the predictions and a
        dictionary mapping an object idx (starting from 1) to a dictionary describing the detected object:
        - `polygon` key : list, the coordinates of the points of the polygon,
        - `confidence` key : float, confidence of the model,
        - `channel` key : str, the name of the predicted class.
    """

    # # # Load the model and get its classes, classes_colors and the model
    # Model = load_model(model)

    # Make a prediction with the model
    detected_polygons, probabilities, mask, overlap = MODELS[model].model.predict(
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
    for channel in range(1, len(MODELS[model].classes)):
        for i, polygon in enumerate(detected_polygons[channel]):
            # Draw the polygons on the image copy.
            # Loop through the class_colors list (channel 1 has color 0)
            ImageDraw.Draw(img2).polygon(
                polygon["polygon"], fill=MODELS[model].colors[channel - 1]
            )

            # Build the dictionary
            # Add an index to dictionary keys to differentiate predictions of the same class
            predict.append(
                {
                    # The list of coordinates of the points of the polygon.
                    # Cast to list of np.int32 to make it JSON-serializable
                    "polygon": np.asarray(polygon["polygon"], dtype=np.int32).tolist(),
                    # Confidence that the model predicts the polygon in the right place
                    "confidence": polygon["confidence"],
                    # The channel on which the polygon is predicted
                    "channel": MODELS[model].classes[channel],
                }
            )

    # Return the blend of the images and the dictionary formatted in json
    return Image.blend(image, img2, 0.5), json.dumps(predict, indent=20)


def get_model_idx(dropdown):
    """
    Get the model selected in dropdown component and return its id

    :param dropdown: A model selected in dropdown
    :return: The model id selected
    """
    return models_name.index(dropdown)


def update_model(model_name):
    """
    Update MODELS dict

    :param model_name: The name of the current model
    """
    MODELS[model_name] = load_model(model_name)


def update_model_title(model_name):
    """
    Update the model title to the title of the current model

    :param model_id: The id of the current model
    :return: A new title
    """
    return f"## {MODELS[model_name].title}"


def update_model_description(model_name):
    """
    Update the model description to the description of the current model

    :param model_id: The id of the current model
    :return: A new description
    """
    return MODELS[model_name].description


with gr.Blocks() as process_image:
    # Create app title
    title = gr.Markdown(f"# {config['title']}")

    # Create app description
    description = gr.Markdown(config["description"])

    # Create dropdown button
    model_name = gr.Dropdown(models_name, value=models_name[0], label="Models")

    load_model(model_name.value)

    model_title = gr.Markdown(f"## {MODELS[model_name.value].title}")
    # model_title = gr.Markdown(f"## {config['models'][model_id.value]['title']}")

    # Create model description
    model_description = gr.Markdown(MODELS[model_name.value].description)

    model_name.change(update_model, model_name)

    # Change model title when the model_id is update
    model_name.change(update_model_title, model_name, model_title)

    # Change model description when the model_id is update
    model_name.change(update_model_description, model_name, model_description)

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
                # Generate example images that can be used as input image for every model
                gr.Examples(config["examples"], inputs=image)

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

    # Clear button: set default values to inputs and output objects
    clear_button.click(
        lambda: (None, None, None),
        inputs=[],
        outputs=[image, image_output, json_output],
    )

    # Create the button to submit the prediction
    submit_button.click(
        query_image, inputs=[model_name, image], outputs=[image_output, json_output]
    )

# Launch the application with the public mode (True or False)
process_image.launch(share=args.public)
