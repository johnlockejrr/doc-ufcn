# Doc-UFCN

Code to train and test Doc-UFCN model.


## Introduction

The Doc-UFCN tool is split into three parts:

- The code to train the model on a given datasets;
- The code to predict the segmentation of images according to the trained model;
- The code to evaluate the model based on the predictions.

A csv configuration file allows to run a batch of experiments at once and also to train, predict or evaluate on combined datasets by only specifying the paths to the datasets folders.

## Preparing the environment

First of all, one needs an environment to run the three experiments presented before. Create a new environment and install the needed packages:

```
$ virtualenv -p python3 doc-ufcn
$ source doc-ufcn/bin/activate
$ pip install -r requirements.txt
```

## Preparing the data

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
In addition, the evaluation is run over json files containing the polygons coordinates that should be in the `labels_json` folders.

## Preparing the configuration files

### `experiments_config.json`

Different files must be updated according to the task one want to run. Since we can run multiple experiments at once, the first configuration file `experiments_config.json` allows to specify the common parameters to use for all the experiments:

| Parameter        | Description                                                                                                        | Default value                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| `classes_names`  | List with the names of the classes / **must be in the same order** as the colors defined in `classes_colors` field | `["background", "text_line"]` |
| `classes_colors` | List with the color codes of the classes                                                                           | `[[0, 0, 0]], [0, 0, 255]]`   |
| `img_size`       | Network input size / **must be the same** as the one used during the label generation                              | `768`                         |
| `no_of_epochs`   | Number of epochs to train the models                                                                               | `200`                         |
| `batch_size`     | Size of batchs to use during training                                                                              | `4`                           |
| `min_cc`         | Threshold to use when removing of small connected components                                                       | `50`                          |
| `save_image`     | List with the sets ["train", "val", "test"] for which we want to save the predicted masks                          | `["val", "test"]`             |
| `omniboard`      | Whether to use Omniboard observer                                                                                  | `false`                       |
| `use_amp`        | Whether to use Automatic Mixed Precision during training                                                           | `false`                       |

Automatic Mixed Precision allows to speed up to training while using less memory (possibility to increase the batch size).

### `experiments.csv`

In the root directory, one has to create an `experiments.csv` file (see `example_experiments.csv`). It contains the experiments names as well as the paths to the datasets and parameters used to continue a training or to fine-tune a model.

| Parameter         | Description                                                                                      | Default value / example                            |
| ----------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------- |
| `experiment_name` | Name of the experiment                                                                           |                                                    |
| `steps`           | List of steps to run ["normalization_params", "train", "prediction", "evaluation"]               | `normalization_params;train;prediction;evaluation` |
| `train`           | Paths to the training datasets                                                                   | `path_to_dataset1;path_to_dataset_2`               |
| `val`             | Paths to the validation datasets                                                                 | `path_to_dataset1;path_to_dataset_2`               |
| `test`            | Paths to the evaluation datasets                                                                 | `path_to_dataset1;path_to_dataset_2`               |
| `restore_model`   | Name of a saved model to resume or fine-tune a training                                          |                                                    |
| `loss`            | Whether to use an initial loss (`initial`) or the best (`best`) saved loss of the restored model | `initial`                                          |

Note: All the steps are dependant, e.g to run the `"prediction"` step, one **needs** the results of the `"normalization_params"` and `"train"` steps.

#### Example

The `example_experiments.csv` file shows an example on how to build the experiments csv file.

| experiment_name | steps                                              | train                                 | val                                   | test                                  | restore_model    | loss      |
| --------------- | -------------------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ---------------- | --------- |
| exp1            | `normalization_params;train;prediction;evaluation` | `~/data/DLA/dataset1;~/data/dataset2` | `~/data/DLA/dataset1;~/data/dataset2` | `~/data/DLA/dataset3;~/data/dataset2` |                  |           |
| exp1            | `prediction;evaluation`                            |                                       |                                       | `~/data/DLA/dataset4`                 |                  |           |
| exp1            | `train;prediction;evaluation`                      | `~/data/DLA/dataset4`                 | `~/data/DLA/dataset4`                 | `~/data/DLA/dataset4`                 | `last_model.pth` | `initial` |
| exp2            | `normalization_params;train;prediction;evaluation` | `~/data/DLA/dataset5`                 | `~/data/DLA/dataset5`                 | `~/data/DLA/dataset5`                 |                  |           |

The first line will start a standard training on two datasets (dataset1 and dataset2) and will be tested also on two datasets (dataset2 and dataset3).

The second line will use the model trained during the first experiment (same experiment_name) and only test it on another dataset (dataset4).

The third line will also use the first trained model (same experiment_name) but will fine-tune it on dataset4. `restore_model` indicates which model to fine-tune and `loss` indicates that the loss should be initialized (datasets used to fine-tune are not the same as the one used for first training).

The last line will run a standard new training on dataset5.

## Start an experiment

To start the experiments:

```
$ bash run_dla_experiment.sh -c experiments.csv
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
$ bash run_dla_experiment.sh -c experiments.csv -s false
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
- Put `"omniboard": true` in `experiments_config.json` file;
- Update the user and password in `run_experiment.py` (l. 34).

## Result of an experiment

The logs of an experiment are saved in `DLA_train.log` file.

Once a model has been trained, it can be found in `./runs/experiment_name/model.pth`.

The predictions are in `./runs/experiment_name/predictions`.

The evaluation results are in `./runs/experiment_name/results`.

## Resume a training

There is no need to re-run the `"normalization_params"` step.
