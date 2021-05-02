#argparser: to be edited by user to customize code
import argparse

def create_parser():
    #mobilenet parser
    parser = argparse.ArgumentParser(description="Thyroid CNN Project Pytorch Model Code")
    parser.add_argument("--model_dir", type=str, default='_thyroid_weighted_focal_extralinear_3rowstack_trainonly', help="folder to save your trained model")
    parser.add_argument("--model_path", type=str, default='', help="actual path to save your trained model")
    parser.add_argument("--pretrained_dir", type=str, help="folder to save pretrained model weights for setup_model function")
    parser.add_argument("--imgpath", type=str, default="/path/to/imgs.hdf5", help="path to img hdf5 file")  #CHANGE if required
    parser.add_argument("--maskpath", type=str, default="/path/to/masks.hdf5", help="path to mask hdf5 file")  #CHANGE if required
    parser.add_argument("--labelpath", type=str, default="/path/to/labels.csv", help="path to labels csv file") #CHANGE if required
    parser.add_argument("--project_home_dir", type=str, default="/your/home/directory/", help="home project directory") #CHANGE
    
    return parser


def create_transformer_parser():
    parser = argparse.ArgumentParser(description="Thyroid Transformer Project Pytorch Model Code")
    parser.add_argument("--model_dir", type=str, default='_thyroid_weighted_focal_extralinear_3rowstack_trainonly', help="folder to save your trained model")
    parser.add_argument("--model_path", type=str, default='', help="actual path to save your trained model")
    parser.add_argument("--pretrained_dir", type=str, help="folder to get pretrained model weights from in setup_model function")
    parser.add_argument("--imgpath", type=str, default="/path/to/imgs.hdf5", help="path to img hdf5 file")  #CHANGE if required
    parser.add_argument("--maskpath", type=str, default="/path/to/masks.hdf5", help="path to mask hdf5 file")  #CHANGE if required
    parser.add_argument("--labelpath", type=str, default="/path/to/labels.csv", help="path to labels csv file") #CHANGE if required
    parser.add_argument("--features2dpath", type=str, default="/path/to/manual2dfeatures.csv", help="path to manual 2d features csv file") #CHANGE if required
    parser.add_argument("--project_home_dir", type=str, default="/your/home/directory/", help="home project directory") #CHANGE

    return parser
