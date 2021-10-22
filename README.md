# Doc-UFCN

This library contains a public implementation of Doc-UFCN, a fully convolutional network presented in the paper "*Multiple Document Datasets Pre-training Improves Text Line Detection With Deep Neural Networks*". This paper is available at https://teklia.com/research/publications/boillet2020/. This library has been developed by the original authors from Teklia (https://teklia.com/).

The model is designed to run various Document Layout Analysis (DLA) tasks like the text line detection or page segmentation.

[Model schema](/resources/UFCN.png)

This library can be used by anyone that has an already trained Doc-UFCN model and want to apply it easily on document images. With only a few lines of code, the trained model is loaded, applied to an image and the detected objects along with some visualizations are obtained.

### Getting started

To use Doc-UFCN in your own scripts, install it using pip:

```
pip install teklia-doc-ufcn
```

### Usage

To apply Doc-UFCN to an image, first add a few imports and load an image. Note that the image should be in RGB.
```
import cv2
from doc_ufcn.main import DocUFCN

image = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
```

Then you can initialize and load the trained model with the parameters used during training. The number of classes should include the background that must have been put as the first channel during training.
```
nb_of_classes = 2
mean = [0, 0, 0]
std = [1, 1, 1]
input_size = 768
model_path = "trained_model.pth"

model = DocUFCN(nb_of_classes, input_size, 'cpu')
model.load(model_path, mean, std)
```

To run the inference on a GPU, you can replace `cpu` by the name of your GPU. In the end, run the prediction:
```
detected_polygons = model.predict(image)
```

### Output

When running inference on an image, the detected objects are returned as in the following example. The objects belonging to a class (except for the background class) are returned as a list containing the confidence score and the polygon coordinates of each object.
```
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

In addition, one can directly retrieve the raw probabilities output by the model using `raw_output=True`. A torch tensor of size `(nb_of_classes, height, width)` is then returned and can be used for further processing.

Lastly, two visualizations can be returned by the model:
  * A mask of the detected objects;
  * An overlap of the detected objects on the input image.

[Mask of detected objects](/resources/mask.png)
[Overlap with the detected objects](/resources/overlap.png)

### Cite us!

If you want to cite us in one of your works, please use the following citation.
```
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
