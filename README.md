# Doc-UFCN

This Python 3 library contains a public implementation of Doc-UFCN, a fully convolutional network presented in the paper [Multiple Document Datasets Pre-training Improves Text Line Detection With Deep Neural Networks](https://teklia.com/research/publications/boillet2020/). This library has been developed by the original authors from [Teklia](https://teklia.com).

The model is designed to run various Document Layout Analysis (DLA) tasks like the text line detection or page segmentation.

![Model schema](https://gitlab.com/teklia/doc-ufcn/-/raw/main/resources/UFCN.png)

This library can be used by anyone that has an already trained Doc-UFCN model and want to easily apply it to document images. With only a few lines of code, the trained model is loaded, applied to an image and the detected objects along with some visualizations are obtained.

### Getting started

To use Doc-UFCN in your own scripts, install it using pip:

```console
pip install doc-ufcn
```

### Usage

To apply Doc-UFCN to an image, one need to first add a few imports and to load an image. Note that the image should be in RGB.
```python
import cv2
from doc_ufcn.main import DocUFCN

image = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
```

Then one can initialize and load the trained model with the parameters used during training. The number of classes should include the background that must have been put as the first channel during training.
```python
nb_of_classes = 2
mean = [0, 0, 0]
std = [1, 1, 1]
input_size = 768
model_path = "trained_model.pth"

model = DocUFCN(nb_of_classes, input_size, 'cpu')
model.load(model_path, mean, std)
```

To run the inference on a GPU, one can replace `cpu` by the name of the GPU. In the end, one can run the prediction:
```python
detected_polygons = model.predict(image)
```

### Output

When running inference on an image, the detected objects are returned as in the following example. The objects belonging to a class (except for the background class) are returned as a list containing the confidence score and the polygon coordinates of each object.
```json
{
  1: [
    {
      'confidence': 0.99,
      'polygon': [(490, 140), (490, 1596), (2866, 1598), (2870, 140)]
    }
    ...
  ],
  ...
}
```

In addition, one can directly retrieve the raw probabilities output by the model using `model.predict(image, raw_output=True)`. A tensor of size `(nb_of_classes, height, width)` is then returned along with the polygons and can be used for further processing.

Lastly, two visualizations can be returned by the model:
  * A mask of the detected objects `mask_output=True`;
  * An overlap of the detected objects on the input image `overlap_output=True`.


By default, only the detected polygons are returned, to return the four outputs, one can use:
```
detected_polygons, probabilities, mask, overlap = model.predict(image, raw_output, mask_output, overlap_output)
```

![Mask of detected objects](https://gitlab.com/teklia/doc-ufcn/-/raw/main/resources/mask.png)
![Overlap with the detected objects](https://gitlab.com/teklia/doc-ufcn/-/raw/main/resources/overlap.png)

### Cite us!

If you want to cite us in one of your works, please use the following citation.
```latex
@inproceedings{boillet2020,
    author = {Boillet, Mélodie and Kermorvant, Christopher and Paquet, Thierry},
    title = {{Multiple Document Datasets Pre-training Improves Text Line Detection With Deep Neural Networks}},
    booktitle = {2020 25th International Conference on Pattern Recognition (ICPR)},
    year = {2021},
    month = Jan,
    pages = {2134-2141},
    doi = {10.1109/ICPR48806.2021.9412447}
}
```

### License

This library is under the 3-Clause BSD License.
