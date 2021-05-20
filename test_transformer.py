from cnn_data_augmentations import *
import transformer_dataset #DatasetPaddedTransformer
from transformer_model import TransformerModel
import model_setup #has setup_model and save_networks functions

from transformer_main import transfconfigwrite, transfconfigread

import numpy as np
import os
from os import path
import operator
import pandas as pd
import glob
import random
from torch import Tensor
import time
import csv
from csv import writer
from csv import reader

import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.utils import resample

from numpy import sqrt
from numpy import argmax
import matplotlib.pyplot as plt

import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms 

import torch.optim as optim
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
from sklearn.utils import resample
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve



#def test_transformer(min_epoch):
def test_transformer(imgpath, maskpath, labelpath, saved_model_dir, project_home_dir, min_epoch):
    config = transfconfigread()
    
    ttotal = 0
    tcorrect = 0   
    
    #DATA
    transformer_test_set = transformer_dataset.DatasetPaddedTransformer("test", config['cvphase'], config['features2d'], config['vertconcat'], config['frametype'], project_home_dir)

    if(config['features2d']):
        if(config['vertconcat']):
            tmodel = TransformerModel(256, use_position_enc = config['pos_encoding'])#256 = number of features in each
        else:
            tmodel = TransformerModel(256+102, use_position_enc = config['pos_encoding'])#256+102 = number of features in each
    else:
        tmodel = TransformerModel(256, use_position_enc = config['pos_encoding'])
    
    phase = "test"
    print("PRETRAINED DIR:", saved_model_dir)

    for param in tmodel.parameters():
        param.requires_grad = False

    #moves to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmodel = tmodel.to(device)
    model_setup.setup_model(tmodel, "test", config['cvphase'], saved_model_dir, saved_model_dir, project_home_dir, min_epoch) 
    
    #########
    #CHECK WHICH PARAM SETUP MODEL USES IN TEST PHASE!
    #########
    
    
    test_set_loader = DataLoader(dataset=transformer_test_set, num_workers=0, batch_size=1, shuffle=False)

    test_all_labels = []
    test_all_probs_ones = []
    test_all_patients = []

    ttotal = 0
    tcorrect = 0

    tmodel.eval()
    #calculate val accuracy
    with torch.no_grad():
        print("num iterations of test:", len(test_set_loader), "but stopping at", config['numpatstest'], "total patients")
        for i, dataa in enumerate(test_set_loader):

            if(i >= (config['numpatstest']-1)):
                print("test DONE at batch number:", i, "and ending now")
                break

            inputs = dataa['input'].to(device)
            labels = dataa['label'].to(device)
            annot_ids = dataa['annot_id']
            
            labels = labels.squeeze(0)
            annot_ids = annot_ids.squeeze(0)
            
            # Forward pass only to get logits/output
            outputs = tmodel(inputs)
            outputs = outputs.squeeze(1)
            
            b = True if True in np.isnan(inputs.cpu().numpy()) else False
            if(b):
                print("STOP!!! NAN IN inputs")
                print("inputs", inputs.cpu().numpy(),"outputs",outputs.detach().cpu().numpy())
            
            if (i % 100 == 0):
                print("in test:", i, inputs.shape, labels.shape, annot_ids.shape)

            # Get predictions from the maximum value                        
            if (config['probFunc'] == 'Softmax'):
                sf = nn.Softmax(dim=1) #makes items in a row add to 1; dim = 0 makes items in a column add to 1
                outputs = sf(outputs)

            _, predicted = torch.max(outputs, 1)

            if torch.cuda.is_available():
                outs_ones = outputs.detach().cpu().numpy()[:, 1]
                labelsnp = labels.cpu().numpy()
            else:
                outs_ones = outputs.detach().numpy()[:, 1]
                labelsnp = labels.numpy()

            test_all_labels = np.append(test_all_labels, labelsnp)
            test_all_probs_ones = np.append(test_all_probs_ones, outs_ones)

            annot_ids = np.asarray(annot_ids)
            test_all_patients = np.append(test_all_patients, annot_ids)

            # Total number of labels
            try:
                ttotal += len(predicted.cpu())
            except:
                print("len 0 labels", labels)
            accuracy = 100 * tcorrect // ttotal
            
            
        #WRITE OUTPUT PROBABILITIES, LABELS for EVERY INSTANCE to CSV FILE
        testsaveallfile = "transformer_test_all_outs" + str(config['cvphase']) + ".csv"
        f=open(testsaveallfile,'w', newline ='\n')
        count = 0
        f.write("annot_id, label, probability\n") #titles
        for i,j,k in zip(test_all_patients, test_all_labels, test_all_probs_ones):
            if (count % 200 == 0):
                print(i, j, k)
            f.write(str(i)) #annot_id
            f.write("," + str(j)) #label
            f.write("," + str(k))
            f.write("\n")
            count += 1
        f.close()