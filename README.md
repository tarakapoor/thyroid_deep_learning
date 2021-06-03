# thyroid_deep_learning

**HOW TO RUN**

Edit the parser arguments in cnn_main.py and transformer_main.py with your own home directory and paths to images, labels and masks.

cnn_main.py: Run this file to train and test CNN (Mobilenet-v2) model and output CNN model predictions to cnn_test_all_outs[cvphase].csv

transformer_main.py: Run this file to train and test Transformer model. It runs CNN model using the lowest validation loss pretrained epoch (saved to configtest.yaml) to extract features, then trains and tests Transformer model with extracted features and outputs overall model predictions to transformer_test_all_outs[cvphase].csv

**SETUP**

HARDWARE: (used to develop models)
GPU Tesla T4
CPU Intel Xeon model 79 ???

SOFTWARE (python package requirements are listed in requirements.txt):
OS: Debian GNU/Linux 10
CUDA Version 11.0
Pytorch 1.6.0

additional libraries:
- albumentations
