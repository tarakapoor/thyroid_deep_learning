# pytorch mobilenet model train
# by tara kapoor

import glob
import random
import argparse
import time

#importing files now
import config #config.py
import parsing_file #either do "from __ import functionname" and just use functionname, or "import ___" and then do ___.functionname
parser = parsing_file.create_parser() 
args = parser.parse_args("")
print("Home directory argument in my code:{}".format(args.project_home_dir))

from mobilenet_dataset import *

from cnn_data_augmentations import * #augmentations are not functions so not sure what this is doing
import train_cnn #train_model (for cnn)
import test_cnn #test_model
from analyze_model_outputs import * #analyze_test_outputs, plot_test_stats, calc_test_stats, bootstrap_auc

def main():
    print(config.phase, "config.phase from config file")
    config.phase = "test"
    print(config.phase, "config.phase from config file")
    config.phase = "train"
    print(config.phase, "config.phase from config file")
    
    #test out dataset functions (don't need if calling in dataloader)
    h5py.File(args.imgpath).keys()
    colnames = ['Labels for each frame', 'Annot_IDs', 'size_A', 'size_B', 'size_C', 'location_r_l_', 'study_dttm', 'age', 'sex', 'final_diagnoses', 'ePAD ID', 'foldNum']
    

    #first round training
    config.trainval = False #use 3/5 of data for train, with val
    config.best_epoch = train_cnn.train_model(args)

    #second round training (4/5 of data)
    config.trainval = True #use 4/5 of data for train, no val
    print(config.best_epoch)
    print("config num epochs", config.num_epochs)
    realtrain = train_cnn.train_model(args) #ignore this returned epoch
    print("\n\n4/5 data model done training!!!")
    print("config num epochs after trainval", config.num_epochs) #should be modified during trainval

    #load test data
    savedepochs = [72, 60, 88, 94, 61] #already knew saved best_epochs so using these, otherwise use config.best_epoch from train phase
    config.best_epoch = savedepochs[config.cvphase]
    print(config.best_epoch, "\nTEST: cv phase", config.cvphase)
    test_cnn.test_model(args, config.best_epoch)

    #analyze test outputs
    testsaveallfile = "cnn_test_all_outs" + str(config.cvphase) + ".csv"
    analyze_test_outputs(testsaveallfile, "cnn")

    
if __name__ == "__main__":
    # execute only if run as a script
    main()
