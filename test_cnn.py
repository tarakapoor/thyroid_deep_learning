from mobilenet_dataset import *
import model_setup #has setup_model and save_networks functions
from cnn_data_augmentations import *

from main import configwrite, configread

import numpy as np
import os
import time

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


def test_model(imgpath, maskpath, labelpath, project_home_dir, min_epoch):
    
    config = configread()
    
    ttotal = 0
    tcorrect = 0
    
    test_set = DatasetThyroid3StackedNew(imgpath, maskpath, labelpath, project_home_dir, "test", config['cvphase'], config['frametype'], transform=transformNorm)
    
    if(config['modeltype'] == "mobilenet"):
        tmodel = models.mobilenet_v2(pretrained=True)
    config = configwrite('phase', "test")
    print(min_epoch)

    #make sure using trainval model
    model_dir = config['modeltype'] + '_thyroid_weighted_focal_extralinear_adj'
    model_path = project_home_dir + 'model/' + model_dir + '/'
    
    print("pretrained dir:", model_dir)
    pretrained_dir = model_dir

    for param in tmodel.parameters():
        param.requires_grad = False

    #modify actual last linear layer in the mobilenet
    tmodel.classifier._modules['1'] = nn.Linear(1280, 256)
    tmodel.classifier._modules['2'] = nn.Linear(256, 2)
    
    #moves to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmodel = tmodel.to(device)
    model_setup.setup_model(tmodel, "test", config['cvphase'], model_dir, pretrained_dir, project_home_dir, min_epoch)

    test_set_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=config['batchSize'], shuffle=False)

    test_all_labels = []
    test_all_probs_ones = []
    test_all_patients = []

    total = 0
    correct = 0

    tmodel.eval() #test mode

    with torch.no_grad():
        for i, data in enumerate(test_set_loader):
            inputs = data['input']

            inputs = data['input'].to(device)
            labels = data['label'].to(device)
            annot_ids = data['annot_id']

            # Forward pass only to get logits/output
            outputs = tmodel(inputs)
            labels = labels.squeeze()

            #apply sigmoid or softmax
            if (config['probFunc'] == 'Softmax'):
                sf = nn.Softmax(dim=1) #makes items in a row add to 1; dim = 0 makes items in a column add to 1
                outputs = sf(outputs)
                if (i == 10):
                    print('SOFTMAX DONE ON TEST')

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs, 1)

            #add labels + predictions to full set for auroc later
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

            if(i%100 == 0):
                try:
                    print(len(test_all_patients))
                    print("test labels vs. PROBABILITIES:", labels.cpu(), outs_ones)#predicted.cpu())
                    for l in range(len(labels)):
                        if (labels[l] == 1):
                            if (predicted[l] == 1):
                                print("yay! predicted 1 correctly")
                                print(labels[l], predicted[l])
                            break
                except:
                    print("allpatients sdfkjlf", "len 0 tlabels...", labels)

            #  USE GPU FOR MODEL
            # Total correct predictions
            try:
                if torch.cuda.is_available():
                    tcorrect += (predicted.cpu() == labels.cpu()).sum()#(thresh_predicted == tlabels.cpu()).sum()
                else:
                    tcorrect += (predicted == labels).sum()
            except:
                print("??")
                tcorrect += (predicted == labelsnp).sum()

            # Total number of labels
            try:
                ttotal += len(predicted.cpu())#labels.size(0)
            except:
                print("len 0 labels", labels)
            accuracy = 100 * tcorrect // ttotal
            
    #WRITE OUTPUT PROBABILITIES, LABELS for EVERY INSTANCE to CSV FILE
    testsaveallfile = "cnn_test_all_outs" + str(config['cvphase']) + ".csv"
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
