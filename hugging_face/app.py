# -*- coding: utf-8 -*-

import gradio as gr
from PIL import Image, ImageDraw

from doc_ufcn import models
from doc_ufcn.main import DocUFCN
from hugging_face.hugging_face_config import get_config

config = get_config("hugging_face/config.json")

model_path, parameters = models.download_model(config["model_name"])

model = DocUFCN(len(parameters["classes"]), parameters["input_size"], "cpu")
model.load(model_path, parameters["mean"], parameters["std"])


def query_image(image):
    detected_polygons, probabilities, mask, overlap = model.predict(
        image, raw_output=True, mask_output=True, overlap_output=True
    )

    image = Image.fromarray(image)
    img2 = image.copy()

    for prediction in detected_polygons[1]:
        draw = ImageDraw.Draw(img2)
        draw.polygon(prediction["polygon"], fill="green")

    image_predicted = Image.blend(image, img2, 0.5)
    return image_predicted


process_image = gr.Interface(
    query_image,
    inputs=[gr.Image()],
    outputs="image",
    title=config["title"],
    description=config["description"],
    examples=config["examples"],
)

process_image.launch(share=True)
