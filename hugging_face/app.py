import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from hugging_face.config import parse_configurations
from hugging_face.tools import UFCNModel

# Parse the CLI arguments
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

# Create a list of models name
models_name = list(MODELS)


def load_model(model_name) -> UFCNModel:
    """
    Retrieve the model, and load its parameters/files if it wasn't done before.

    :param model_name: The name of the selected model
    :return: The UFCNModel instance selected
    """
    assert model_name in MODELS
    model = MODELS[model_name]
    # Load the model's files if it wasn't done before
    if not model.loaded:
        model.load()
    return model


def query_image(model_name: gr.Dropdown, image: gr.Image) -> list([Image, json]):
    """
    Loads a model and draws the predicted polygons with the color provided by the model on an image

    :param model: A model selected in dropdown
    :param image: An image to predict
    :return: Image and dict, an image with the predictions and a
        dictionary mapping an object idx (starting from 1) to a dictionary describing the detected object:
        - `polygon` key : list, the coordinates of the points of the polygon,
        - `confidence` key : float, confidence of the model,
        - `channel` key : str, the name of the predicted class.
    """

    # Load the model and get its classes, classes_colors and the model
    ufcn_model = load_model(model_name)

    # Make a prediction with the model
    detected_polygons, probabilities, mask, overlap = ufcn_model.model.predict(
        input_image=image, raw_output=True, mask_output=False, overlap_output=False
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
        for polygon in detected_polygons[channel]:
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

    :param model_name: The name of the selected model
    :return: A new title
    """
    return f"## {MODELS[model_name].title}", MODELS[model_name].description


with gr.Blocks() as process_image:
    # Create app title
    gr.Markdown(f"# {config['title']}")

    # Create app description
    gr.Markdown(config["description"])

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
            # Create a column so that the JSON output doesn't take the full size of the page
            # Create a collapsible region
            with gr.Row(), gr.Column(), gr.Accordion("JSON"):
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
