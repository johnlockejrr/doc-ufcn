# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from hugging_face.config import parse_configurations
from hugging_face.tools import UFCNModel

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

models_name = [model["model_name"] for model in config["models"]]

<<<<<<< HEAD
# Check that the paths of the examples are valid
for example in config["examples"]:
    assert Path.exists(
        Path(example)
    ), f"The path of the image '{example}' does not exist."

# Cached models, maps model_name to UFCNModel object
MODELS = {
    model["model_name"]: UFCNModel(
        name=model["model_name"],
        colors=model["classes_colors"],
        title=model["title"],
        description=model["description"],
    )
    for model in config["models"]
}
=======
def load_model(model_name, model_id):
    # Download the model
    model_path, parameters = models.download_model(name=model_name)

    # Store classes_colors list
    classes_colors = config["models"][model_id]["classes_colors"]
>>>>>>> 9a64357... Add a new config and a selector of models

# Create a list of models name
models_name = list(MODELS)

<<<<<<< HEAD

def load_model(model_name: str) -> UFCNModel:
    """
    Load a model by name if it doesn't already exist then return the model

    :param model_name: The name of the selected model
    :return: The UFCNModel instance selected
    """
    assert model_name in MODELS
    model = MODELS[model_name]
    if not model.loaded:
        model.load()
    return model

=======
    # Check that the number of colors is equal to the number of classes -1
    assert len(classes)-1 == len(
        classes_colors
    ), f"The parameter classes_colors was filled with the wrong number of colors. {len(classes)-1} colors are expected instead of {len(classes_colors)}."
    
>>>>>>> 9a64357... Add a new config and a selector of models
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

    return classes, classes_colors, model

def query_image(model_name: gr.Dropdown, image: gr.Image) -> list([Image, json]):
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
    # Get the id of the model in the list of models
    model_id = models_name.index(dropdown)

    classes, classes_colors, model = load_model(dropdown, model_id)

    # Load the model and get its classes, classes_colors and the model
    ufcn_model = load_model(model_name)

    # Make a prediction with the model
    detected_polygons, probabilities, mask, overlap = ufcn_model.model.predict(
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
    for channel in range(1, ufcn_model.num_channels):
        for i, polygon in enumerate(detected_polygons[channel]):
            # Draw the polygons on the image copy.
            # Loop through the class_colors list (channel 1 has color 0)
            ImageDraw.Draw(img2).polygon(
                polygon["polygon"], fill=ufcn_model.colors[channel - 1]
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
                    "channel": ufcn_model.classes[channel],
                }
            )

    # Return the blend of the images and the dictionary formatted in json
    return Image.blend(image, img2, 0.5), json.dumps(predict, indent=2)


def update_model(model_name: gr.Dropdown) -> str:
    """
    Update the model title to the title of the current model

    :param model_name: A model selected in dropdown
    :return: A new title
    """
    return f"## {MODELS[model_name].title}", MODELS[model_name].description

def get_value(dropdown):
    return models_name.index(dropdown.value)

with gr.Blocks() as process_image:

    # Create app title
    title = gr.Markdown(f"# {config['title']}")

    # Create app description
    description = gr.Markdown(config["description"])
<<<<<<< HEAD

    # Create dropdown button
    model_name = gr.Dropdown(models_name, value=models_name[0], label="Models")

    # get models
    selected_model: UFCNModel = MODELS[model_name.value]

    # Create model title
    model_title = gr.Markdown(f"## {selected_model.title}")

    # Create model description
    model_description = gr.Markdown(selected_model.description)

    # Change model title and description when the model_id is update
    model_name.change(update_model, model_name, [model_title, model_description])
=======
>>>>>>> 9a64357... Add a new config and a selector of models

    dropdown = gr.Dropdown(models_name, value=models_name[0], label="Models")

    model_id = get_value(dropdown)

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
            with gr.Row():
                # Generates an output image that does not support upload
                image_output = gr.Image(interactive=False)

            # Create a row under the predicted image
            with gr.Row():
                # Create a column so that the JSON output doesn't take the full size of the page
                with gr.Column():
                    # # Create a collapsible region
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
