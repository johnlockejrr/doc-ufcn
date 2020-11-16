# U-FCN

Code to train and test U-FCN model.


## Introduction

The U-FCN tool is split into three parts:

- The code to train the model on a given dataset;
- The code to predict the segmentation of testing images according to the trained model;
- The code to evaluate the model based on the previous predictions.

## Preparing the environment

First of all, one needs an environment to run the three experiments presented before. Create a new environment and install the needed packages:

```
$ virtualenv -p python3 ufcn
$ source ufcn/bin/activate
$ pip install -r requirements.txt
```

### Preparing the data

To train and test the model, all the images and their annotations must be in the `./data` folder following this hierarchy:

```
.
├── data
│   ├── classes.txt
│   ├── test
│   │   ├── images
│   │   └── labels
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── val
│       ├── images
│       └── labels
├── ...
```

Once the images are in the right directories, one has to compute the normalization parameters (mean value and standard deviation) of the training set:

```
$ python normalization_params.py with img_size=XXX
```

The *img_size* variable is the size of the images on which the model will be trained (typically `img_size=768`).

## Training U-FCN

### Configuration files

Different files must be updated according to the task one want to run. In the `./utils` directory, one has the first configuration file for the training stage named `training_config.json`. One can change the different parameters of the training like the number of epochs or the batch size directly in this file. One has to add the mean values and standard deviations computed before to allow the normalization of the images during training. The *img_size* chosen before also has to be put in this configuration file.

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

These colors must be the same than the one used in the annotation images.

### Start and visualize the training

To start the training :

```
$ python train.py with utils/training_config.json
```

One can see the training progress using Tensorboard. In a new terminal:

```
$ tensorboard --logdir ./runs/experiment_name
```

The model and the useful file for visualization are stored in `./runs/experiment_name`.

One can also log and visualize the training using [Omniboard](https://github.com/vivekratnavel/omniboard) and need to add a MongoObserver (l. 35):

```
ex.observers.append(MongoObserver(
    url='mongodb://username:password@omniboard/dbname'))
```

## Prediction and evaluation

### Configuration file

The configuration file `./utils/testing_config.json` is used for the prediction and the evaluation steps. In this file, the background must be defined as the other classes. We also have to add the *normalization_params* used during training as well as the *img_size* parameter and the *experiment_name*.

Once a model has been trained, it can be found in `./runs/experiment_name`.

### Start the prediction

To start the prediction:

```
$ python predict.py with utils/testing_config.json
```

The predicted labels are put in `./runs/experiment_name/predictions`.

### Metrics

We chose to compute the Mean Intersection-over-Union (IoU) metric to evaluate the performance of the models but also the pixel precision, recall and F1-score. A value is computed for each class of each image. We then compute the average of a particular over all the images.

In addition, some object metrics are computed for different IoU thresholds: precision, recall, F1-score and AP.

### Start the evaluation

To start the evaluation:

```
$ python evaluate.py with utils/testing_config.json
```

The summary of the computed metrics is in `./runs/experiment_name/res/Results.json` along with some plots:
- Precision-recall curve for each class;
- F-score vs. confidence score;
- Precision vs. confidence score;
- Recall vs. confidence score.
