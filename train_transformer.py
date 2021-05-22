from cnn_data_augmentations import *
import model_setup #has setup_model and save_networks functions
from transformer_main import transfconfigwrite, transfconfigread
import analyze_model_outputs

import transformer_dataset #DatasetPaddedTransformer
from transformer_model import TransformerModel

import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms 

import torch.optim as optim
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
from sklearn.utils import resample

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



def train_transformer(project_home_dir):
    """Train Transformer model on train features and validate on val features.
    Also trainval mode to train on both train and validation data until the lowest validation loss epoch from first round of training.
    In train phase, call plot_test_stats and calc_test_stats to calculate probability prediction thresholds.
    Return lowest validation loss epoch and directory of saved model.
    """
    config = transfconfigread()
    
    patientwise_auroc = 0
    #learning rate
    lrs = []
    
    seed_value = 1
    torch.manual_seed(seed_value)

    losses = []
    #f_losses = []
    losses_val = []
    #f_losses_val = []
    
    running_loss = 0.0
    running_loss_val = 0.0

    total_all = 0
    correct_all = 0

    transall_labels = []
    all_probs_ones = []
    transall_patients = [] #val
    train_transall_patients = [] #train
    
    tlabelsnp = np.zeros(16)
    
    epoch_aurocs = []

    #default for early stopping
    min_val_loss = 10
    prev_val_loss = 10
    epochs_no_improve = 0
    n_epochs_stop = 5
    early_stop = False
    min_epoch = 0

    ###############
    
    #max epochs
    if(config['trainval'] == True):
        config = transfconfigwrite('num_epochs', config['best_epoch']+2)
    else:
        config = transfconfigwrite('num_epochs', 100) #default train 100 epochs to find which has lowest val loss
    print("\n\nTraining transformer for", config['num_epochs'], "epochs")
    
    #############DONE WITH DEFINING VARIABLES
    ###############NOW DEFINE MODEL
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    training_directory = "{0}".format(project_home_dir)
    os.chdir(training_directory)


    if not (config['frametype'] == "singleframe"):
        config = transfconfigwrite('pos_encoding', False) #just making sure
    print("positional encoding?", config['pos_encoding'], "\n\n")
    
    if(config['features2d']):
        if(config['vertconcat']):
            model = TransformerModel(256, use_position_enc= config['pos_encoding'])
        else:
            model = TransformerModel(256+102, use_position_enc= config['pos_encoding'])#number of features in each.
    else:
        model = TransformerModel(256, use_position_enc= config['pos_encoding'])
    for param in model.parameters():
        param.requires_grad = True
        
    print("batchsize: {}, learning rate: {}, output classes: {}, optimizer: SGD".format(config['batchSize'], config['lr'], config['num_classes']))
    
    model_dir = config['modeltype'] + 'TRYtransformer_thyroid_no_focalX'
        
    if(config['lr'] != 0.001):
        model_dir = model_dir + '_lr' + str(int(config['lr']*1000))
    
    if(config['weightdecay']==False):
        model_dir = model_dir + "_no_decay"
    
    if(config['frametype'] == "adjacent"):
        model_dir = model_dir + "_adj"
    elif(config['frametype'] == "singleframe"):
        model_dir = model_dir + "_singleframe"
        if(config['pos_encoding']):
            model_dir = model_dir + "_posenc"
    
    model_path = project_home_dir + 'model/' + model_dir + '/'
    print("dirs:", model_path, model_dir)
    
    #moves to gpu
    model = model.to(device)
    config = transfconfigwrite('phase', "train")                           
    model_setup.setup_model(model, "train", config['cvphase'], model_dir, model_dir, project_home_dir, config['best_epoch'])
    print(config['modeltype'], "Model Construction Complete")

    #adding weights to loss function because of imbalance in dataset
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    if(config['weightdecay']):
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay = 0.002, momentum=0.9)
    
    ####loading DATA
    if(not config['trainval']):
        transformer_train_set = transformer_dataset.DatasetPaddedTransformer("train", config['cvphase'], config['features2d'], config['vertconcat'], config['frametype'], project_home_dir)
    else:
        transformer_trainval_set = transformer_dataset.DatasetPaddedTransformer("trainval", config['cvphase'], config['features2d'], config['vertconcat'], config['frametype'], project_home_dir)
    transformer_val_set = transformer_dataset.DatasetPaddedTransformer("val", config['cvphase'], config['features2d'], config['vertconcat'], config['frametype'], project_home_dir)
    
    config = transfconfigread() #file was updated during datasetpaddedtransformer function
            
    #learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config['lr'], epochs=config['num_epochs'], steps_per_epoch=config['numpatstrain'])
    
    #START TRAINING!
    for epoch in range(config['num_epochs']):
        #DOING THIS BY EPOCH FOR AUROC
        transall_labels = []
        all_probs_ones = []
        transall_patients = []
        train_transall_patients = []

        epochstart = time.time()
        print("\n\nEPOCH", epoch, ":\n")
        correct = 0
        total = 0
        traincorrect = 0
        traintotal = 0
        valcount = 0.0
        traincount = 0.0

        if(config['trainval']):
            train_set_loader = DataLoader(dataset=transformer_trainval_set, num_workers=0, batch_size=config['batchSize'], shuffle=False)
            print('Loading total', len(transformer_trainval_set), 'training images--------')
        else:
            train_set_loader = DataLoader(dataset=transformer_train_set, num_workers=0, batch_size=config['batchSize'], shuffle=False)
            print('Loading total', len(transformer_train_set), 'training images--------')

        val_set_loader = DataLoader(dataset=transformer_val_set, num_workers=0, batch_size=config['batchSize'], shuffle=False)
        print('Loading total', len(transformer_val_set), 'val images--------')
        
#        UPDATE LEARNING RATE ONCE PER EPOCH
        if(epoch>0):
            scheduler.step()
            print("Learning rate at epoch", epoch, "is:", scheduler.get_last_lr(), "aka", optimizer.param_groups[0]['lr'])
        lrs.append(optimizer.param_groups[0]['lr'])
        
    
        #this calls getitem (for each i in train_set_loader)
        print("num iterations of train:", len(train_set_loader), "but stopping at", config['numpatstrain'], "total patients")
        model.train() #train mode
        for i, data in enumerate(train_set_loader):
            if(i >= (config['numpatstrain']-1)):
                print("DONE at batch number:", i, "and ending now")
                break
            
            inputs = data['input'].to(device)
            labels = data['label'].to(device)
            annot_ids = data['annot_id']
            framenums = data['framenum']
            
            optimizer.zero_grad()
            #get batch size out of all
            labels = labels.squeeze(0)
            annot_ids = annot_ids.squeeze(0)
            
            train_transall_patients = np.append(train_transall_patients, annot_ids[0])
            if (i % 100 == 0):
                print("in train:", inputs.shape, labels.shape, annot_ids.shape)
                print("batch index {}, 0/1 distribution: {}/{}".format(i, len(np.where(labels.cpu().numpy() == 0)[0]),
            len(np.where(labels.cpu().numpy() == 1)[0])))
                
            # forward + backward + optimize (to find good parameters: weights + bias)
            outputs = model(inputs).to(device)
            #now squeeze outt he 1 from the outputs shape [x, 1, 2]
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            traincount += 1.0
            
            if (config['probFunc'] == 'Softmax'):
                sf = nn.Softmax(dim=1) #makes items in a row add to 1; dim = 0 makes items in a column add to 1
                outputs = sf(outputs)

            _, trainpredicted = torch.max(outputs, 1)
            try:
                if torch.cuda.is_available():
                    traincorrect += (trainpredicted.cpu() == labels.cpu()).sum()
                else:
                    traincorrect += (trainpredicted == labels).sum()
            except:
                print("??")
                traincorrect += (trainpredicted == labelsnp).sum()

            # Total number of labels
            try:
                traintotal += len(labels.cpu())
            except:
                print(i, "index len 0 trainlabels; input shape", np.shape(inputs.cpu().numpy()))

            trainaccuracy = 100 * traincorrect // traintotal

            
            x = 40
            if (i % x == 0):
                #validation phase of model
                model.eval()
                #calculate val accuracy
                with torch.no_grad():
                    print("num iterations of val:", len(val_set_loader), "but stopping at", config['numpatsval'], "total patients")
                    for j, dataa in enumerate(val_set_loader):
                        if(j >= (config['numpatsval']-1)):
                            print("val DONE at batch number:", j, "and ending now")
                            break
                        
                        tinputs = dataa['input'].to(device)
                        tlabels = dataa['label'].to(device)
                        tannot_ids = dataa['annot_id']

                        tlabels = tlabels.squeeze(0)
                        tannot_ids = tannot_ids.squeeze(0)
            
                        # Forward pass only to get logits/output
                        outs = model(tinputs)
                        outs = outs.squeeze(1)

                        # Get predictions from the maximum value                        
                        if (config['probFunc'] == 'Softmax'):
                            sf = nn.Softmax(dim=1) #makes items in a row add to 1; dim = 0 makes items in a column add to 1
                            outs = sf(outs)
                        elif (config['probFunc'] == 'Sigmoid'):
                            sf = nn.Sigmoid()
                            outs = sf(outs)

                        _, predicted = torch.max(outs, 1)

                        #add labels + predictions to full set for auroc later
                        if torch.cuda.is_available():
                            outs_ones = outs.detach().cpu().numpy()[:, 1]
                            tlabelsnp = tlabels.cpu().numpy()
                            transall_labels = np.append(transall_labels, tlabelsnp)
                            
                            if(j%100 == 0 or (tlabelsnp[0] == "1" and j%20 == 0)):
                                if(np.isnan(outs.detach().cpu().numpy()).any()):
                                    print("val inputs:", tinputs.detach().cpu().numpy())
                                    print("val outputs:", outs.detach().cpu().numpy(), outs_ones, "vs. labels:", tlabelsnp)
                            
                        else:
                            outs_ones = outs.detach().numpy()[:, 1]
                            tlabels = tlabels.numpy()
                            transall_labels = np.append(transall_labels, tlabels)

                        all_probs_ones = np.append(all_probs_ones, outs_ones)

                        tannot_ids = np.asarray(tannot_ids)
                        transall_patients = np.append(transall_patients, tannot_ids)


                        if(j%50 == 0):
                            try:
                                #val loss
                                loss_val = criterion(outs, tlabels)

                                running_loss_val += loss_val.item()
                                valcount += 1
                            except:
                                print("val loss failed\n\n")
                            
                        #  USE GPU FOR MODEL
                        # Total correct predictions
                        try:
                            if torch.cuda.is_available():
                                correct += (predicted.cpu() == tlabels.cpu()).sum()
                            else:
                                correct += (predicted == tlabels).sum()
                        except:
                            print("??")
                            correct += (predicted == tlabelsnp).sum()

                        # Total number of labels
                        try:
                            total += len(tlabels.cpu())
                        except:
                            print(j, "index len 0 tlabels; input shape", np.shape(tinputs.cpu().numpy()))
                        accuracy = 100 * correct // total
                        #end of torch no grad

                #print statistics
                message = 'epoch: %d, iters: %d, time: %.3f, with loss: %.5f, val loss: %.5f, train_acc: %.4f, val accuracy: %.3f' % (
                    epoch, i + 1, time.time() - epochstart, loss, loss_val, trainaccuracy.data, accuracy.data)
                print(message)

                model.train() #back to train mode, end of validation mode

        total_all += total
        correct_all += correct

        losses.append(running_loss / traincount)

        val_loss = running_loss_val / valcount
        losses_val.append(val_loss)
        
        print("Epoch", epoch, "overall val loss:", val_loss, "overall train loss:", (running_loss / traincount)), #"or train over len trainset?", (running_loss / len(train_set)))
        if (epoch % 30 == 0):
            print("all patient ids train:", train_transall_patients, "\n and val:", transall_patients[0], transall_patients[10], transall_patients[300])
        
        if (not config['trainval']):
            if (val_loss < prev_val_loss): #current epoch val loss improved
          # Save the model #torch.save(model)
                epochs_no_improve = 0
                if ((val_loss < min_val_loss)):
                    min_val_loss = val_loss
                    min_epoch = epoch
                prev_val_loss = val_loss
                early_stop = False
            else:
                print("val loss did not improve vs. last epoch", prev_val_loss, ", epochs not improving =", epochs_no_improve, "... min is", min_val_loss, "from epoch", min_epoch)
                epochs_no_improve += 1
                prev_val_loss = val_loss

            if (epoch > 48 and epochs_no_improve >= n_epochs_stop):
                print('Early stopping! end of epoch', epoch, ", val loss =", val_loss, "min val loss =", min_val_loss, "from epoch", min_epoch)
                early_stop = True

        running_loss = 0.0
        running_loss_val = 0.0
        valcount = 0.0
        traincount = 0.0

        patients = [] #distinct patients
        for p in range(len(transall_patients)):
            if not (transall_patients[p] in patients):
                patients.append(transall_patients[p])

        patient_ave_preds = []
        patientlabels = []
        count = 0
        sum_pat_pred = 0
        cur_pat_labels = []
        patientwise_auroc = 0

        while(count < len(patients)):
            for p in range(len(transall_patients)):
                if (transall_patients[p] == patients[count]): #one patient at a time
                    sum_pat_pred += all_probs_ones[p]
                    cur_pat_label = transall_labels[p]
                    cur_pat_labels.append(transall_labels[p])
            patient_ave_preds.append(sum_pat_pred / float(len(cur_pat_labels)))
            patientlabels.append(cur_pat_labels[0])
            count += 1
            cur_pat_labels = []
            sum_pat_pred = 0
            
        #calculate auroc based on average score for each patient
        print(patientlabels, patient_ave_preds)
        patientwise_auroc = roc_auc_score(patientlabels, patient_ave_preds)
        print("Epoch", epoch,"AUROC by patient-wise average predictions =", patientwise_auroc)
        epoch_aurocs.append(patientwise_auroc)

        if early_stop:
            print("Training Stopped")
            break

        save_freq = 30
        if (config['trainval']):
            if (epoch % save_freq == 0 or epoch >= (config['best_epoch']-2)):
                print('saving the latest model (epoch %d) of learning rate %f' % (epoch, config['lr']))
                print(model_path, model_dir)
                model_setup.save_networks(model, epoch, config['cvphase'], model_path, model_dir)

        
    patients = []
    for p in range(len(transall_patients)):
        if not (transall_patients[p] in patients):
            patients.append(transall_patients[p])
    print("distinct patients:", patients)

    patient_ave_preds = []
    patientlabels = []
    count = 0
    sum_pat_pred = 0
    cur_pat_labels = []
    patientwise_auroc = 0

    while(count < len(patients)):
        for p in range(len(transall_patients)):
            if (transall_patients[p] == patients[count]): #one patient at a time
                sum_pat_pred += all_probs_ones[p]
                cur_pat_label = transall_labels[p]
                cur_pat_labels.append(transall_labels[p])

        patient_ave_preds.append(sum_pat_pred / float(len(cur_pat_labels)))
        patientlabels.append(cur_pat_labels[0])
        count += 1
        cur_pat_labels = []
        sum_pat_pred = 0


    if(not config['trainval']):
        print("Min validation loss epoch:", min_epoch, ", loss =", min_val_loss, ", train loss =", losses[min_epoch], ", val auroc =", epoch_aurocs[min_epoch])
    else:
        print("Min train 3/5 validation loss epoch:", config['best_epoch'], ", loss =", min_val_loss)
    
    analyze_model_outputs.plot_test_stats(losses, losses, losses_val, losses_val, epoch_aurocs, patientlabels, patient_ave_preds, "transformer")
    analyze_model_outputs.calc_test_stats(patients, patientlabels, patient_ave_preds)
    
    return min_epoch, model_dir
