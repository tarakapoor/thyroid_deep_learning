# thyroid_deep_learning

**How to run:
**
Edit the parser arguments in cnn_main.py with your own home directory and paths to images, labels and masks.

Run cnn_main.py to train and test CNN (Mobilenet-v2) model and output CNN model predictions to cnn_test_all_outs[cvphase].csv

Run transformer_main.py to run CNN model at the lowest validation loss pretrained epoch (saved to configtest.yaml) to extract features, then train and test Transformer model and output overall model predictions to transformer_test_all_outs[cvphase].csv


