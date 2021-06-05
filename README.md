# CNN+Transformer Model

**ABOUT**
The CNN+Transformer architecture consists of a MobileNetv2 convolutional neural network (CNN) for feature extraction concatenated with a bi-layer Transformer-Encoder network.

![figure2](https://user-images.githubusercontent.com/44348827/120879105-f825bb80-c575-11eb-935d-330fbcb9f16a.png)

**FILES**

configtest.yaml:
Contains pre-set variables for CNN model development (e.g. learning rate, batch size) for model as well as variables updated during model training automatically (e.g. probability prediction threshold, lowest validation loss epoch). Do not manually update.

configtransformer.yaml:
Contains pre-set variables for Transformer model development (e.g. learning rate, batch size) for model as well as variables updated during model training automatically (e.g. probability prediction threshold, lowest validation loss epoch). Do not manually update.

--------
mobilenet_preprocess.py
Contains functions (crop_bounding_box, transform_and_crop_new, transform_and_crop_largest) for image preprocessing: cropping images around masks and resizing to 224x224.

mobilenet_dataset.py
Create CNN model dataset. Function load_datasets_new for loading CNN inputs: sorting images into cross validation folds and stacking 3 images at a time into input instances along with corresponding labels and patient IDs.
DataLoader (DatasetThyroid3StackedNew class) calls functions from **mobilenet_preprocess** file to load images, then feeds them into load_datasets_new or load_datasets_single_frame function to get CNN inputs.

train_cnn.py
Train CNN model. Called twice by **cnn_main** file: once to train on only train data and determine lowest validation loss epoch (trainval false); then to train until that epoch on training and validation data (trainval true).
Setup MobileNetv2 model with all layers unfrozen.
Create dataloader using DatasetThyroid3StackedNew class from **mobilenet_dataset** file.
Update configtest.yaml file with lowest validation loss epoch and optimal probability prediction thresholds based on training outputs (normal, weighting true positive rate, weighting false positive rate).
Save model weights using save_networks function from **model_setup** file when in trainval true phase.

test_cnn.py
Test CNN model.
Load trained CNN model using setup_model function from **model_setup** file.
Run on test data and output predictions to cnn_test_all_outs[cvphase].csv by calling analyze_test_outputs function from **analyze_model_outputs** file.

cnn_main.py
Run entire data preprocessing, training and testing of CNN model and output of model predictions to cnn_test_all_outs[cvphase].csv



cnn_feature_extraction.py
Create function to load (from lowest validation loss epoch, saved to configtest.yaml file) and run trained CNN model (exclude final classification layer) on train, validation and test data to extract feature vectors to feed into Transformer model. 

transformer_model.py:
Creates the custom Transformer architecture (TransformerModel class). The Transformer architecture consists of a bi-layer Transformer-Encoder followed by a fully-connected classification layer. Each Transformer encoder layer contains 2-head encoders, comprised of self-attention and feedforward sub-layers for classification. The Transformer encoder layer input is of size (S,N,E), where S is the sequence length of the number of feature vectors per patient (36), N is batch size of 1 per mini-batch, and E is the length (256) of each feature vector.

transformer_dataset.py
Create Transformer model dataset. Function load_csv_padded_transformer for loading CNN feature vectors and grouping by patient. Functions available for appending manually extracted 2D features either horizontally or vertically to CNN extracted features.
DataLoader (load_csv_padded_transformer class) calls load_csv_padded_transformer or function equivalent for adding manual features to get Transformer input vectors.

transformer_main.py
Run entire CNN feature vector extraction, training and testing of Transformer model and output of model predictions to transformer_test_all_outs[cvphase].csv

model_setup.py
Function setup_model for creating directories for new model and loading pretrained model weights if in test phase.
Function save_networks for saving model weights.

analyze_model_outputs.py
Functions calc_test_stats and plot_test_stats.
Function analyze_test_outputs for calculating AUROC and other stats and saving model outputs to csv files.

**HOW TO RUN**

Edit the parser arguments in cnn_main.py and transformer_main.py with your own home directory and paths to images, labels and masks.

cnn_main.py: Run this file to train and test CNN (Mobilenet-v2) model and output CNN model predictions to cnn_test_all_outs[cvphase].csv

transformer_main.py: Run this file to train and test Transformer model.

**SETUP**

HARDWARE: (used to develop models)
GPU Tesla T4
CPU Intel Xeon model 79 ???

SOFTWARE (python package requirements are listed in requirements.txt):
OS: Debian GNU/Linux 10
CUDA Version 11.0
Pytorch 1.6.0

Additional libraries:
- albumentations
