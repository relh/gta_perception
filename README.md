# ROB535 Team 9 Perception Repo

This repository contains code to classify images for the Perception portion of ROB535.

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
