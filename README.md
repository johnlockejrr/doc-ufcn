# U-FCN

Code to train and test U-FCN model.


## Introduction

The U-FCN tool is split into three parts:

- The code to train the model on a given dataset;
- The code to predict the segmentation of images according to the trained model;
- The code to evaluate the model based on the predictions.

## Preparing the environment

First of all, one needs an environment to run the three experiments presented before. Create a new environment and install the needed packages:

```
$ virtualenv -p python3 ufcn
$ source ufcn/bin/activate
$ pip install -r requirements.txt
```

## Preparing the data

To train and test the model, all the images and their annotations must be in the `./data` folder following this hierarchy:

```
.
├── data
│   ├── classes.txt
│   ├── test
│   │   └── images
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── val
│       ├── images
│       └── labels
├── ...
```

The labels should be generated directly at the network input size (*img_size*) to avoid resizing (that can cause mergings of regions).

## Preparing the configuration files

### `config.json`

Different files must be updated according to the task one want to run. In the root directory, one has the first configuration file `config.json`. One can change the different parameters like the number of epochs or the batch size directly in this file according to the following:

| Parameter         | Description                                                                                                          | Default value                                                   |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `experiment_name` | Name of the experiment                                                                                               | `ufcn`                                                          |
| `classes_names`   | List with the names of the classes / **must be in the same order** as the colors defined in the `./data/classes.txt` | `["background", "text_line"]`                                   |
| `img_size`        | Network input size / **must be the same** as the one used during the label generation                                | `768`                                                           |
| `no_of_epochs`    | Number of epochs to train the model                                                                                  | `200`                                                           |
| `batch_size`      | Size of batchs to use during training                                                                                | `2`                                                             |
| `min_cc`          | Threshold to use when removing of small connected components                                                         | `50`                                                            |
| `save_image`      | List with the sets ["train", "val", "test"] for which we want to save the predicted masks.                           | `["test"]`                                                      |
| `steps`           | List with the steps to run ["normalization_params", "train", "prediction", "evaluation"]                             | `["normalization_params", "train", "prediction", "evaluation"]` |
| `omniboard`       | Whether to use Omniboard observer                                                                                    | `false`                                                         |
| `restore_model`   | Path to the last saved model to resume training                                                                      | `None`                                                                |

Note: All the steps are dependant, e.g to run the `"prediction"` step, one **needs** the results of the `"normalization_params"` and `"train"` steps.

### `data/classes.txt`

In addition, one has to update the `./data/classes.txt` file. This text file declares the different classes used during the experiment. Each line must contain a single color code (RGB format) corresponding to a class. Here are presented two examples of contents that can be put in `classes.txt`. In this file, the background (black) must be defined by adding on the first line `0 0 0`.

For a 2-classes line segmentation:

```
0 0 0
0 0 255
```

For a global segmentation containing different classes:

```
0 0 0
255 255 255
0 255 0
0 150 0
```

Note: These colors must be the same than the ones in the annotation images and be defined in the same order as in `classes_names` of `config.json` file.

Note 2: One **must not** add an empty line at the end of the file.

## Start an experiment

To start the experiment and run the steps defined in `config.json` file at once:

```
$ bash run_dla_experiment.sh
```

There's a way to be notified in slack when training has finished (successfully or not):
- Create a webhook here https://my.slack.com/services/new/incoming-webhook/;
- Save the webhook key into `~/.notify-slack-cfg` (looks like: `T02TKKSAX/B246MJ6HX/WXt2BWPfNhSKxdoFNFblczW9`)
- Make sure that the notifier is working:
```
python notify-slack.py "WARN: notifier works"
```
- The slack notification is used by default;
- To start the experiment without this slack notification run:
```
$ bash run_dla_experiment.sh -s false
```

## Follow a training

### Tensorboard

One can see the training progress using Tensorboard. In a new terminal:

```
$ tensorboard --logdir ./runs/experiment_name
```

The model and the useful file for visualization are stored in `./runs/experiment_name`.

### Omniboard

One can also log and visualize the training using [Omniboard](https://github.com/vivekratnavel/omniboard):
- Put `"omniboard": true` in `config.json` file;
- Update the user and password in `run_experiment.py` (l. 23).

## Result of an experiment

The logs of an experiment are saved in `DLA_train.log` file.

Once a model has been trained, it can be found in `./runs/experiment_name/model.pth`.

The predictions are in `./runs/experiment_name/predictions`.

**So far, no evaluation, come in a few days.**

## Resume a training

To resume a training, one can add `"resume_training": "path/to/last_model.pth"` to the `config.json` file. If the training images are the same as the first one used, there is no need to re-run the `"normalization_params"` step.
