# Doc-UFCN

This Python 3 library contains a public implementation of Doc-UFCN, a fully convolutional network presented in the paper [Multiple Document Datasets Pre-training Improves Text Line Detection With Deep Neural Networks](https://teklia.com/research/publications/boillet2020/). This library has been developed by the original authors from [Teklia](https://teklia.com).

The model is designed to run various Document Layout Analysis (DLA) tasks like the text line detection or page segmentation.

![Model schema](https://gitlab.com/teklia/dla/doc-ufcn/-/raw/main/resources/UFCN.png)

This library can be used to train a model, fine-tune a model or directly apply a trained Doc-UFCN model to document images.

## Getting started

To use Doc-UFCN in your own scripts, install it using pip:

```shell
pip install doc-ufcn
```

## Model inference

With only a few lines of code, the trained model is loaded, applied to an image and the detected objects along with some visualizations are obtained.

### Usage

To apply Doc-UFCN to an image, one need to first add a few imports (optionally, set the logging config to make logs appear on stdout) and to load an image. Note that the image should be in RGB.
```python
import cv2
import logging
import sys
from doc_ufcn.main import DocUFCN

logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
    level=logging.INFO
)

image = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
```

Then one can initialize and load a trained model with the parameters used during training. The number of classes should include the background that must have been put as the first channel during training. By default, the model is loaded in evaluation mode. To load it in training mode, use `mode="train"`.
```python
nb_of_classes = 2
mean = [0, 0, 0]
std = [1, 1, 1]
input_size = 768
model_path = "trained_model.pth"

model = DocUFCN(nb_of_classes, input_size, 'cpu')
model.load(model_path, mean, std, mode="eval")
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
  * A mask of the detected objects, using `mask_output=True` parameter;
  * An overlap of the detected objects on the input image, using `overlap_output=True` parameter.


By default, only the detected polygons are returned, to return the four outputs, one can use:
```python
detected_polygons, probabilities, mask, overlap = model.predict(
    image, raw_output=True, mask_output=True, overlap_output=True
)
```

![Mask of detected objects](https://gitlab.com/teklia/dla/doc-ufcn/-/raw/main/resources/mask.png)
![Overlap with the detected objects](https://gitlab.com/teklia/dla/doc-ufcn/-/raw/main/resources/overlap.png)

### Models

We provide various open-source models, stored on [HuggingFace](https://huggingface.co/Teklia) and every model prefixed by `doc-ufcn-` is supported. For example, to download our [generic page detection model](https://huggingface.co/Teklia/doc-ufcn-generic-page) and load it, one can use:
```python
from doc_ufcn import models
from doc_ufcn.main import DocUFCN

model_path, parameters = models.download_model('generic-page')

model = DocUFCN(len(parameters['classes']), parameters['input_size'], 'cpu')
model.load(model_path, parameters['mean'], parameters['std'])
```

By default, the most recent version of the model will be downloaded. One can also use a specific version using the following line:
```python
model_path, parameters = models.download_model('generic-page', version="main")
```

## Training

The Doc-UFCN tool is split into three parts:

- The code to train the model on given datasets;
- The code to predict the segmentation of images according to the trained model;
- The code to evaluate the model based on the predictions.

A csv configuration file allows to run a batch of experiments at once and also to train, predict or evaluate on combined datasets by only specifying the paths to the datasets folders.

### Preparing the environment

First of all, one needs an environment to run the three experiments presented before. Create a new environment and install the needed packages:
```shell
pip install doc-ufcn[training]
```

### Preparing the data

To train and test the model, all the images and their annotations of a dataset should be in a folder following this hierarchy:

```
.
├── dataset_name
│   ├── test
│   │   ├── images
│   │   └── labels_json
│   ├── train
│   │   ├── images
│   │   ├── labels
│   │   └── labels_json
│   └── val
│       ├── images
│       ├── labels
│       └── labels_json
├── ...
```

The labels should be generated directly at the network input size (*img_size*) to avoid resizing (that can cause mergings of regions).
In addition, the evaluation is run over json files containing the original polygons coordinates that should be in the `labels_json` folders.

### Preparing the configuration files

#### `experiments_config.json`

Different files must be updated according to the task one want to run. Since we can run multiple experiments at once, the first configuration file `experiments_config.json` allows to specify the common parameters to use for all the experiments:

| Parameter        | Description                                                                                                        | Default value                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| `classes_names`  | List with the names of the classes / **must be in the same order** as the colors defined in `classes_colors` field | `["background", "text_line"]` |
| `classes_colors` | List with the color codes of the classes                                                                           | `[[0, 0, 0]], [0, 0, 255]]`   |
| `img_size`       | Network input size / **must be the same** as the one used during the label generation                              | `768`                         |
| `no_of_epochs`   | Number of epochs to train the models                                                                               | `200`                         |
| `batch_size`     | Size of batchs to use during training                                                                              | None                          |
| `no_of_params`   | Maximum number of parameters supported by the CPU/GPU                                                              | None                          |
| `bin_size`       | Size between two groups of images.                                                                                 | 20                            |
| `min_cc`         | Threshold to use when removing of small connected components                                                       | `50`                          |
| `save_image`     | List with the sets ["train", "val", "test"] for which we want to save the predicted masks                          | `["val", "test"]`             |
| `use_amp`        | Whether to use Automatic Mixed Precision during training                                                           | `false`                       |

The background class **must** always be defined at the first position in the `classes_names` and `classes_colors` fields.

Automatic Mixed Precision allows to speed up the training while using less memory (possibility to increase the batch size). Either the batch size or the number of parameters should be defined.

#### `experiments.csv`

In the root directory, one has to create an `experiments.csv` file (see `example_experiments.csv`). It contains the experiments names as well as the paths to the datasets and parameters used to continue a training or to fine-tune a model.

| Parameter         | Description                                                                                      | Default value / example                            |
| ----------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------- |
| `experiment_name` | Name of the experiment                                                                           |                                                    |
| `steps`           | List of steps to run ["normalization_params", "train", "prediction", "evaluation"]               | `normalization_params;train;prediction;evaluation` |
| `train`           | Paths to the training datasets                                                                   | `path_to_dataset1;path_to_dataset_2`               |
| `val`             | Paths to the validation datasets                                                                 | `path_to_dataset1;path_to_dataset_2`               |
| `test`            | Paths to the evaluation datasets                                                                 | `path_to_dataset1;path_to_dataset_2`               |
| `restore_model`   | Name of a saved model to resume or fine-tune a training                                          |                                                    |
| `same_classes`    | Whether the classes of the current experiment are the same as those of the model to resume       | `True`                                             |
| `loss`            | Whether to use an initial loss (`initial`) or the best (`best`) saved loss of the restored model | `initial`                                          |

Note: All the steps are dependent, e.g to run the `"prediction"` step, one **needs** the results of the `"normalization_params"` and `"train"` steps.

##### Example

The `example_experiments.csv` file shows an example on how to build the experiments csv file.

| experiment_name | steps                                              | train                                 | val                                   | test                                  | restore_model    | same_classes | loss      |
| --------------- | -------------------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ---------------- | ------------ | --------- |
| exp1            | `normalization_params;train;prediction;evaluation` | `~/data/DLA/dataset1;~/data/dataset2` | `~/data/DLA/dataset1;~/data/dataset2` | `~/data/DLA/dataset3;~/data/dataset2` |                  |              |           |
| exp1            | `prediction;evaluation`                            |                                       |                                       | `~/data/DLA/dataset4`                 |                  |              |           |
| exp1            | `train;prediction;evaluation`                      | `~/data/DLA/dataset4`                 | `~/data/DLA/dataset4`                 | `~/data/DLA/dataset4`                 | `last_model.pth` |              | `initial` |
| exp2            | `normalization_params;train;prediction;evaluation` | `~/data/DLA/dataset5`                 | `~/data/DLA/dataset5`                 | `~/data/DLA/dataset5`                 |                  |              |           |

The first line will start a standard training on two datasets (dataset1 and dataset2) and will be tested also on two datasets (dataset2 and dataset3).

The second line will use the model trained during the first experiment (same experiment_name) and only test it on another dataset (dataset4).

The third line will also use the first trained model (same experiment_name) but will fine-tune it on dataset4. `restore_model` indicates which model to fine-tune and `loss` indicates that the loss should be initialized (datasets used to fine-tune are not the same as the one used for first training).

The last line will run a standard new training on dataset5.

### Start an experiment

To start the experiment:
```shell
$ ./run_dla_experiment.sh -c experiments.csv
```

There's a way to be notified in slack when training has finished (successfully or not):
- Create a webhook here https://my.slack.com/services/new/incoming-webhook/;
- Save the webhook key into `~/.notify-slack-cfg` (looks like: `T02TKKSAX/B246MJ6HX/WXt2BWPfNhSKxdoFNFblczW9`)
- Make sure that the notifier is working:
```shell
python tools/notify-slack.py "WARN: notifier works"
```
- The slack notification is used by default;
- To start the experiment without this slack notification run:
```shell
$ ./run_dla_experiment.sh -c experiments.csv -s false
```

### Follow a training

#### Tensorboard

One can see the training progress using Tensorboard. In a new terminal:

```shell
$ tensorboard --logdir ./runs/experiment_name
```

The model and the useful file for visualization are stored in `./runs/experiment_name`.


#### MLflow

MLflow logging is also available in Doc-UFCN. Information about the instance and the experiment need to be specified in the configuration file `experiments_config.json`, under the key `mlflow`.

```json
# experiments_config.json
{
  ...
  "mlflow": {
    "experiment_id": ...,
    "run_name": null,
    "tracking_uri": ...,
    "s3_endpoint_url": ...,
    "aws_access_key_id": ...,
    "aws_secret_access_key": ...
  }
}
```

- `experiment_id`: ID of the MLflow experiment where the run will be recorded,
- `run_name`: Optional name of the created run,
- `tracking_uri`: URL towards the MLflow instance, see [MLFLOW_TRACKING_URI](https://www.mlflow.org/docs/latest/quickstart.html?highlight=mlflow_tracking_uri#launch-a-tracking-server-on-a-remote-machine),
- `s3_endpoint_url`: URL towards the MLflow instance's storage, see [MLFLOW_S3_ENDPOINT_URL](https://www.mlflow.org/docs/latest/python_api/mlflow.environment_variables.html?highlight=mlflow_s3_endpoint#mlflow.environment_variables.MLFLOW_S3_ENDPOINT_URL),
- `aws_access_key_id` and `aws_secret_access_key`: [AWS credentials](https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup.html#setup-credentials) if the storage is Amazon S3 or Amazon S3-compatible. Only needed when logging artifacts.

### Result of an experiment

The logs of an experiment are saved in `DLA_train.log` file.

Once a model has been trained, it can be found in `./runs/experiment_name/model.pth`.

The predictions are in `./runs/experiment_name/predictions`.

The evaluation results are in `./runs/experiment_name/results`.

### Resume a training

To resume a training, just by adding epochs for example, one just has to:
  - Remove the `"normalization_params"` step;
  - Indicate the name of the model to resume in `restore_model` parameter;
  - Set the value of the `loss` at `"best"`.

## Model fine-tuning

Several models are available on [HuggingFace](https://huggingface.co/Teklia) for inference but they can also be used as pre-trained weights. It is then possible to fine-tune them on a new task and/or new data.

For this, it is necessary to follow the same data preparation steps as above, as well as the specification of the configuration files. To fine-tune a model, one needs to:
  - Remove the `"normalization_params"` step;
  - Indicate the name of the model to resume in `restore_model` parameter. If the name contains a `.pth` extension, the corresponding local file will be retrieved. Otherwise, to retrieve a model from HuggingFace, one just needs to put the name of the model starting with `doc-ufcn-`;
  - Set the value of the `loss` at `"initial"`;
  - Set the `same_classes` parameter to `False` if the classes used for training the new model are different from the classes used during the pre-training.

Once these parameters have been updated, the training can be started and followed as described above.

## Cite us!

If you want to cite us in one of your works, please use the following citation.
```latex
@inproceedings{boillet2020,
    author = {Boillet, Mélodie and Kermorvant, Christopher and Paquet, Thierry},
    title = {{Multiple Document Datasets Pre-training Improves Text Line Detection With
              Deep Neural Networks}},
    booktitle = {2020 25th International Conference on Pattern Recognition (ICPR)},
    year = {2021},
    month = Jan,
    pages = {2134-2141},
    doi = {10.1109/ICPR48806.2021.9412447}
}
```

## License

This library is under the 3-Clause BSD License.
