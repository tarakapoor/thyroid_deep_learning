# pytorch mobilenet model train
# by tara kapoor

import glob
import random
import argparse
import time

#importing files now
import configparser
import ast

import yaml

def configwrite(varname, value): #for yaml
    """Write new value for variable to config yaml file for CNN.
    Return updated config file.
    Keyword arguments:
    varname -- name of variable in config file
    value -- new value to assign
    """
    with open(r'/home/jupyter/google-cloud-sdk/configtest.yaml') as file:
        dictloaded = yaml.load(file, Loader=yaml.FullLoader)
        
    dictloaded[varname] = value

    with open(r'/home/jupyter/google-cloud-sdk/configtest.yaml', 'w') as file: #add on, not override
        documents = yaml.dump(dictloaded, file)
        
    return configread() #return updated dictionary after writing


def configread():
    """Read config yaml file for CNN.
    Return dictionary of keys and values from config file.
    """
    with open(r'/home/jupyter/google-cloud-sdk/configtest.yaml') as file:
        dictloaded = yaml.load(file, Loader=yaml.FullLoader)
    return dictloaded


parser = argparse.ArgumentParser(description="Thyroid CNN Project Pytorch Model Code")
parser.add_argument("--model_dir", type=str, default='_thyroid_weighted_focal_extralinear_3rowstack_trainonly', help="folder to save your trained model") #set in function
parser.add_argument("--pretrained_dir", type=str, help="folder to save pretrained model weights for setup_model function")
parser.add_argument("--imgpath", type=str, default="/path/to/imgs.hdf5", help="path to img hdf5 file")  #CHANGE if required
parser.add_argument("--maskpath", type=str, default="/path/to/masks.hdf5", help="path to mask hdf5 file")  #CHANGE if required
parser.add_argument("--labelpath", type=str, default="/path/to/labels.csv", help="path to labels csv file") #CHANGE if required
parser.add_argument("--project_home_dir", type=str, default="/your/home/dir", help="home project directory") #CHANGE
args = parser.parse_args("")
print("Home directory argument in my code:{}".format(args.project_home_dir))


import train_cnn #train_model (for cnn)
import test_cnn #test_model
from analyze_model_outputs import analyze_test_outputs #also plot_test_stats, calc_test_stats, bootstrap_auc


def main():
    """Main function for training and testing CNN model and saving outputs.
    Config file is updated automatically within functions in other files."""
    config = configread()
    config = configwrite('phase', "train")

    #first round training
    config = configwrite('trainval', False)
    bestepoch = train_cnn.train_model(args.imgpath, args.maskpath, args.labelpath, args.project_home_dir)
    config = configwrite('best_epoch', bestepoch)
    
    #second round training (4/5 of data)
    config = configwrite('trainval', True) #use 4/5 of data for train, no val
    print(config['best_epoch'])
    print("config num epochs", config['num_epochs'])
    
    realtrain = train_cnn.train_model(args.imgpath, args.maskpath, args.labelpath, args.project_home_dir) #ignore this returned epoch
    print("\n\n4/5 data model done training!!!")
    print("config num epochs after trainval", config['num_epochs']) #should be modified during trainval

    #load test data
    #epochs = [72, 60, 88, 94, 61] #update if you have presaved best epochs
    #config = configwrite('best_epoch', epochs[config['cvphase']])
    print('Best epoch from training:', config['best_epoch'], "\nTEST: CV phase", config['cvphase'])
    test_cnn.test_model(args.imgpath, args.maskpath, args.labelpath, args.project_home_dir, config['best_epoch'])

    config = configread()
    
    #analyze test outputs
    testsaveallfile = "cnn_test_all_outs" + str(config['cvphase']) + ".csv"
    analyze_test_outputs(testsaveallfile, "mobilenet")


    
if __name__ == "__main__":
    # execute only if run as a script
    main()
