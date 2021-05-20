# transformer pytorch model train
# by tara kapoor

import glob
import random
import argparse
import time

#importing files now
import configparser
import ast
import yaml

def transfconfigwrite(varname, value): #for yaml
    with open(r'/home/jupyter/google-cloud-sdk/configtransformer.yaml') as file:
        dictloaded = yaml.load(file, Loader=yaml.FullLoader)
    dictloaded[varname] = value

    with open(r'/home/jupyter/google-cloud-sdk/configtransformer.yaml', 'w') as file: #add on, not override
        documents = yaml.dump(dictloaded, file)
    return transfconfigread() #return updated dictionary after writing


def transfconfigread():
    with open(r'/home/jupyter/google-cloud-sdk/configtransformer.yaml') as file:
        dictloaded = yaml.load(file, Loader=yaml.FullLoader)
    return dictloaded


parser = argparse.ArgumentParser(description="Thyroid Transformer Project Pytorch Model Code")
parser.add_argument("--pretrained_dir", type=str, help="folder to save pretrained model weights for setup_model function")
parser.add_argument("--imgpath", type=str, default="/path/to/imgs.hdf5", help="path to img hdf5 file")  #CHANGE if required
parser.add_argument("--maskpath", type=str, default="/path/to/masks.hdf5", help="path to mask hdf5 file")  #CHANGE if required
parser.add_argument("--labelpath", type=str, default="/path/to/labels.csv", help="path to labels csv file") #CHANGE if required
parser.add_argument("--project_home_dir", type=str, default="/your/home/dir", help="home project directory") #CHANGE
args = parser.parse_args("")
print("Home directory argument in my code:{}".format(args.project_home_dir))


import train_cnn #train_model (for cnn)
import test_cnn #test_model
import train_transformer #train_transformer (for transformer)
import test_transformer #test_transformer
import analyze_model_outputs #analyze_test_outputs, plot_test_stats, calc_test_stats, bootstrap_auc

from main import configread, configwrite

import cnn_feature_extraction

def main():
    
    cnnconfig = configread()
    cnnconfig = configwrite('phase', "test")        

    cnnconfig = configwrite('trainval', False)
    print('Best epoch from training:', cnnconfig['best_epoch'], "\nTEST: CV phase", cnnconfig['cvphase'])
    cnn_feature_extraction.mobilenet_model(args.imgpath, args.maskpath, args.labelpath, args.project_home_dir, cnnconfig['model_dir'], cnnconfig['best_epoch'])
    
    #now extracting trainval + test features
    cnnconfig = configwrite('trainval', True)
    cnn_feature_extraction.mobilenet_model(args.imgpath, args.maskpath, args.labelpath, args.project_home_dir, cnnconfig['model_dir'], cnnconfig['best_epoch'])
    
    #TRANSFORMER
    transfconfig = transfconfigread()
    transfconfig = transfconfigwrite('cvphase', cnnconfig['cvphase'])
    transfconfig = transfconfigwrite('phase', "train")
    
    print("\n\n\ntransfconfig dict loaded")    
    for key, value in transfconfig.items():
        print (key + " : " + str(value))
    
    #first round training
    transfconfig = transfconfigwrite('trainval', False)
    bestepoch, saved_model_dir = train_transformer.train_transformer(args.imgpath, args.maskpath, args.labelpath, args.project_home_dir)
    transfconfig = transfconfigwrite('best_epoch', bestepoch)
    
    #second round training (4/5 of data)
    transfconfig = transfconfigwrite('trainval', True) #use 4/5 of data for train, no val
    print(transfconfig['best_epoch'])
    print("transfconfig num epochs", transfconfig['num_epochs'])
    
    realtrain, saved_model_dir = train_transformer.train_transformer(args.imgpath, args.maskpath, args.labelpath, args.project_home_dir) #ignore this returned epoch
    print("\n\n4/5 data model done training!!!")
    print("transfconfig num epochs after trainval", transfconfig['num_epochs']) #should be modified during trainval

    #load test data
    #epochs = [99, 99, 99, 99, 99]
    #transfconfig = transfconfigwrite('best_epoch', epochs[transfconfig['cvphase']])
    print('Best epoch from training:', transfconfig['best_epoch'], "\nTEST: CV phase", transfconfig['cvphase'])
    test_transformer.test_transformer(args.imgpath, args.maskpath, args.labelpath, saved_model_dir, args.project_home_dir, transfconfig['best_epoch'])
    #analyze test outputs
    testsaveallfile = "transformer_test_all_outs" + str(transfconfig['cvphase']) + ".csv"
    analyze_model_outputs.analyze_test_outputs(testsaveallfile, "transformer")


    
if __name__ == "__main__":
    # execute only if run as a script
    main()
