# ROB535 Team 9 Perception Repo

This repository contains code to classify images for the Perception portion of ROB535.

Currently we finetune pretrained models from the package `cnn_finetune` to our dataset, including data augmentation from other datasets. 

We also have a number of tasks which can be run.

## Getting additional data running
Note: extra data is not stored in folders like our data. It has one folder for all images and one folder for all annotations. Also, annotations are stored in VOC files, and do not have the same 23 class labels as our .bin files. For these reason we load it separately in carnet.py.

1) in /home/ubuntu/more_train/ download either 10k imgs or all 200k imgs (and annotations) from https://fcav.engin.umich.edu/sim-dataset
2) extract them:
$ tar zxvf repro_10k_images.tgz
$ tar zxvf repro_10k_annotations.tgz
3) move annotations and images folders back to /home/ubuntu/more_train/
4) run as usual! May want to experiment with 3 classes vs. all, but seems to help! Appears to be marginally slower (maybe 20% slower per iteration)

## Environment setup
To make a conda environment for our code, run `conda env create -f env535.yml`
Then, `source activate env535.yml` and run `python3 carnet.py` with appropriate args

## Old versions included:

It contains the code for Squeeze and Excitation Networks:
- `baseline.py`
- `se_inception.py`
- `se_module.py`
- `se_resnet.py`
- `utils.py`

It contains demo code provided by the instructors:
- `classes.csv`
- `demo.py`

It contains our custom code.

1) Calculating the mean and standard deviation of the dataset:
- `calc_mean_std.py`
- `mean_std.txt`

2) Running our CarNet model
- `carnet.py`

CarNet is a slightly modified version of the squeeze and excitation ResNet. It contains 4 residual blocks instead of the neighboring 3 and 5.
CarNet is trained to predict which of the 23 object classes an image contains. 

