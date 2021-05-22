import pandas as pd
from PIL import Image
import os
import os.path

import operator
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

import numpy as np
import re
from pathlib import Path
import tables
import cv2
import h5py
import math
import random

#data aug
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch import ToTensor
import albumentations as A

import torch.optim as optim
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
from sklearn.utils import resample

from transformer_main import transfconfigwrite, transfconfigread


def normalize(col):
    """NORMALIZE 2D FEATURES from CSV file (manually extracted).
    Add minimum value to all feature values to make minimum 0, then normalize so all feature values between 0 and 1.
    Input is column of feature values (1 feature value per frame) for a given feature, output is normalized column of feature values."""

    minfeat = np.nanmin(col)
    if(minfeat < 0):
        for row in range(len(col)):
            col[row] = col[row] + (-1*minfeat)
    #min feature should now be 0 (no negatives)
    
    maxfeat = np.nanmax(col)
    if(maxfeat != 0):
        for row in range(len(col)):
            if(math.isnan(col[row])):
                col[row] = 0.0
            col[row] = (col[row]/maxfeat)
    return col    


def process2dfeatures(vertconcat, cvphase, features2dpath):
    """Extract manual 2d features, normalize features by column.
    Keyword arguments:
    vertconcat -- whether to vertically concatenate (true) or horizontally concatenate (false) manual 2d features.
    cvphase -- cross validation fold (0 to 4)
    features2dpath -- path to manual 2d features.
    Return lists of feature vectors by frame, label, patient ID and frame number within patient."""

    if(vertconcat):
        featurelength = 256
    else:
        featurelength = 102
    cur_all_concats = np.zeros((0,featurelength)) #reset to empty
    concatpats = []
    concatlabs = []
    concatframenums = []
    
    if(phase == "train"):
        if(cvphase == 0):
            curfold = 3
        elif(cvphase == 1):
            curfold = 4
        else:
            curfold = cvphase - 2
    elif(phase == "val"):
        if(cvphase == 4):
            curfold = 0
        else:
            curfold = cvphase + 1
    elif(phase == "test"):
        curfold = cvphase
    elif(phase == "trainval"):
        if(cvphase == 0):
            curfold = 3
        elif(cvphase == 1):
            curfold = 4
        else:
            curfold = cvphase - 2

    rowcount = 0
    numberind = 0
    with open(features2dpath, newline='') as infh:
        print("opened csv to concatenate manual features")
        reader = csv.reader(infh)
        rowcount = 0

        mypats = []
        curstartsubtract = 0
        curind = 1
        for row in reader:
            if(row[2] != '0' and row[2] != '1'):
                print("error col label:", row[2])
                continue

            numberind = (row[0].index('_'))+1
            curpatindexfull = row[0]
            concatpats.append(row[1])
            if(row[1] not in mypats):
                mypats.append(row[1])
                curind = 1 #first framenum, renumber
            concatframenums.append(curind)#(curpatindexfull[numberind:])
            concatlabs.append(row[2])

            concats = np.array(row[7:], dtype=float)
            extrapad = np.zeros((1, featurelength-102))
            if(len(concats) != 102):
                print("error patient", row[1], "framenum from csv", curpatindexfull[numberind:], "framenum actual", curind)
                continue
            if(vertconcat):
                if(len(concats)<featurelength):
                    concats = np.append(concats, extrapad) #pad to 256

            cur_all_concats = np.vstack((cur_all_concats, concats)) #should have length 256
            if(rowcount % 5000 == 0):
                print("row:", rowcount, "curind", curind)
            #end loop, now new loop
            rowcount += 1
            curind += 1 #framenum
        print("rowcount", rowcount)

    #loop through columns and normalize
    for colnum in range(len(cur_all_concats[0])):
        cur_all_concats[:, colnum] = normalize(cur_all_concats[:, colnum])
        
    print("concatframenums", np.shape(concatframenums))
    return cur_all_concats, concatpats, concatlabs, concatframenums
        

def load_csv_padded_transformer(phase, cvphase, frametype, project_home_dir): #no padded 2d features    
    """Extract features from CNN csv file for given phase and cvphase, stack based on frametype.
    NO manual 2d features added here.
    Keyword arguments:
    phase -- train, val, trainval or test (which data to use)
    cvphase -- cross validation fold (0 to 4)
    frametype -- adjacent, equalspaced or singleframe (whether to stack frames or not, and how if so)
    Return (for given cross validation fold and train/val/trainval/test phase) lists of all feature vectors, labels, patient IDs and frame numbers within patient."""
    foundcount = 0
    
    config = transfconfigread()
    
    trainvalfile = config['featurestrainvalfile']
    testfile = config['featurestestfile']
    filename = config['featurestrainfile']
    valfile = config['featuresvalfile']
        
    
    curfold = 0
    print("CVPHASE:", cvphase)
    print("csv:", filename)
    if(phase == "train"):
        curfile = filename
        if(cvphase == 0):
            curfold = 3
        elif(cvphase == 1):
            curfold = 4
        else:
            curfold = cvphase - 2
    elif(phase == "val"):
        curfile = valfile
        if(cvphase == 4):
            curfold = 0
        else:
            curfold = cvphase + 1
    elif(phase == "test"):
        curfile = testfile
        curfold = cvphase
    elif(phase == "trainval"):
        curfile = trainvalfile
        if(cvphase == 0):
            curfold = 3
        elif(cvphase == 1):
            curfold = 4
        else:
            curfold = cvphase - 2

    #just cnn features
    featurelength = 256
        
    cur_all_pat_inds = [] #framenum
    cur_all_patients = []
    cur_all_labels = []
    cur_all_probs = np.zeros((0,featurelength))#256))
    
    #col 0 = framenum, col 1 = patient id, col 2 = label, rest = probabilities
    rowcount = 0
    with open(curfile, newline='') as infh:
        print("opened csv for cnn features")
        reader = csv.reader(infh)
        rowcount = 0
        for row in reader:
            if(row[3] == ''):
                print("ERROR", row[2])
                continue
            cur_all_pat_inds.append(row[0])
            cur_all_labels.append(row[2])
            cur_all_patients.append(row[1])
            probs = np.array(row[3:], dtype=float)
            if(rowcount % 500 == 0):
                print("row:", rowcount)
            cur_all_probs = np.vstack((cur_all_probs, probs))
            rowcount += 1
        print("rowcount", rowcount)
            
    print("length of probs, labels, ids", len(cur_all_probs), len(cur_all_labels), len(cur_all_patients))
    
    #get number of images per patient
    distinct_patient_ids = []
    distinct_num_pats = []

    
    #SORT IN ORDER OF FRAME NUM NOW, BEFORE PADDING!
    cur_all_pat_inds = np.array(cur_all_pat_inds).astype(np.float64)
    print("type of cur_all_pat_inds", type(cur_all_pat_inds))
    templist = list(zip(cur_all_patients, cur_all_pat_inds, cur_all_probs, cur_all_labels))
    sortedlist = sorted(templist, key=operator.itemgetter(0, 1))
    cur_all_patients, cur_all_pat_inds, cur_all_probs, cur_all_labels = zip(*sortedlist)
    print("\n\n\n should be sorted (framenums, patient id, label):", cur_all_pat_inds[0:10], cur_all_patients[0:10], cur_all_labels[0:10])
    
    
    print("\n\nLOADING", phase,"PHASE")
    maxpat = ""
    maxpatcount = 0
    curpatcount = 0
    
    for patt in cur_all_patients:
        if (patt not in distinct_patient_ids):
            curpatcount = 0
            distinct_patient_ids.append(patt)
            if(patt == "90_"):
                print(cur_all_patients[:10])
            for xz in range(len(cur_all_patients)):                
                if(cur_all_patients[xz] == patt):
                    curpatcount += 1
                    if(curpatcount >= maxpatcount):
                        maxpat = patt
                        maxpatcount = curpatcount
                        
            distinct_num_pats.append(curpatcount)
            print("patient", patt, ": CNN extracted # of frames =", curpatcount)
    
    print("cur all probs shape", np.shape(cur_all_probs))
    #sort
    tempp = list(zip(distinct_num_pats, distinct_patient_ids)) 
    tempp = sorted(tempp) 
    distinct_num_pats, distinct_patient_ids = zip(*tempp)    
    print(phase, "PATIENTS:", distinct_patient_ids)
    print("cvphase", cvphase,"numpatients input:", len(distinct_patient_ids), " total length:", len(cur_all_patients))    
    
    if(phase == "train" or phase == "trainval"):
        config = transfconfigwrite('numpatstrain', len(distinct_patient_ids))
        print("numpatstrain", config['numpatstrain'])
    elif(phase == "val"):
        config = transfconfigwrite('numpatsval', len(distinct_patient_ids))
        print("numpatstrain", config['numpatsval'])
    elif(phase == "test"):
        config = transfconfigwrite('numpatstest', len(distinct_patient_ids))
        print("numpatstrain", config['numpatstest'])
    
    curpatprobs = np.zeros((0,featurelength))
    curpatlabs = []
    curpatids = []
    curpatframenums = []
    
    allfeatures = np.zeros((0, featurelength))
    alllabels = []
    allids = []
    allframenums = []
    
    #ADD INDIVIDUAL PATIENT VECTORS TO ALLFEATURES VECTOR
    #pad and stack
    counttt = 0
    seq_len = 36

    for patind in range(len(distinct_patient_ids)):
        curpatprobs = np.zeros((0,featurelength))
        curpatlabs = []
        curpatids = []
        curpatframenums = []
        
        pat = distinct_patient_ids[patind]
        for p in range(len(cur_all_patients)):
            if (cur_all_patients[p] == pat):
                curprobs = cur_all_probs[p]
                curprobs = curprobs.reshape((1,featurelength))
                curlabels = cur_all_labels[p]
                curids = cur_all_patients[p]
                curframenums = cur_all_pat_inds[p]
                
                curpatprobs = np.concatenate((curpatprobs, curprobs), axis=0)
                curpatlabs = np.append(curpatlabs, curlabels)
                curpatids = np.append(curpatids, curids)
                curpatframenums = np.append(curpatframenums, curframenums)
                
        numimgs = len(curpatprobs)
        goodlenprobs = np.zeros((seq_len,featurelength))
        #PAD NOW
        if (numimgs > seq_len):
            num_seqs = 1
            lencurseq = 36
            while((numimgs / num_seqs) > seq_len):
                num_seqs += 1
            
            lencurseq = numimgs / num_seqs
            
            for g in range(num_seqs):
                #split into groups of less than 36 vectors of length 256
                start = int(g*lencurseq)
                end = int(((g+1)*lencurseq))
                if(g<(num_seqs)-1):
                    currentseq = curpatprobs[start:end]
                    currentlabs = curpatlabs[start:end]
                    currentids = curpatids[start:end]
                    currentframes = curpatframenums[start:end]
                else:
                    currentseq = curpatprobs[start:]
                    currentlabs = curpatlabs[start:]
                    currentids = curpatids[start:]
                    currentframes = curpatframenums[start:]
                    
                #pad each to 36
                currentseq = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(currentseq), torch.from_numpy(goodlenprobs)], batch_first=True, padding_value=0.0)
                currentseq = np.array(currentseq[0])
                while(len(currentlabs)<seq_len):
                    currentlabs = np.append(currentlabs, currentlabs[0]) #or just '0'?
                    currentids = np.append(currentids, currentids[0])
                    currentframes = np.append(currentframes, 0)
                if((g==0) and (patind == 0)):
                    print("shape of padded sequence, labels, ids:", np.shape(currentseq), np.shape(currentlabs), np.shape(currentids), np.shape(currentframes))
                
                #add 36 at a time from curpat probs to allfeatures
                allfeatures = np.concatenate((allfeatures, currentseq), axis=0)
                alllabels = np.append(alllabels, currentlabs)
                allids = np.append(allids, currentids)
                allframenums = np.append(allframenums, currentframes)
            
        elif (numimgs < seq_len):
            #just pad to 36
            diff = seq_len-len(curpatprobs)
            while(len(curpatlabs)<seq_len):
                curpatlabs = np.append(curpatlabs, curpatlabs[0])
                curpatids = np.append(curpatids, curpatids[0])
                curpatframenums = np.append(curpatframenums, curpatframenums[0])#0)

            currentseq = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(curpatprobs), torch.from_numpy(goodlenprobs)], batch_first=True, padding_value=0.0)
            currentseq = np.array(currentseq[0])
            
            #add the 36 from curpatprobs to allfeatures
            allfeatures = np.concatenate((allfeatures, currentseq), axis=0)
            alllabels = np.append(alllabels, curpatlabs)
            allids = np.append(allids, curpatids)
            allframenums = np.append(allframenums, curpatframenums)

        else:
            allfeatures = np.concatenate((allfeatures, curpatprobs), axis=0)#np.append(allfeatures, curpatprobs)
            alllabels = np.append(alllabels, curpatlabs)
            allids = np.append(allids, curpatids)
            allframenums = np.append(allframenums, curpatframenums)
        
        counttt+= 1
        
        if(phase == "train" or phase == "trainval"):
            config = transfconfigwrite('numpatstrain', int(len(allfeatures) / 36))
        elif(phase == "val"):
            config = transfconfigwrite('numpatsval', int(len(allfeatures) / 36))
        elif(phase == "test"):
            config = transfconfigwrite('numpatstest', int(len(allfeatures) / 36))
        
    print("example features:")
    print(allfeatures[0][0], allfeatures[0][255])
    
    print("\nFINAL features, labels, ids shapes", np.shape(allfeatures), np.shape(alllabels), np.shape(allids), np.shape(allframenums))
    return(allfeatures, alllabels, allids, allframenums) #return feature vectors and labels and patient ids for transformer


def load_csv_padded_transformer_horiz_concat(phase, cvphase, frametype, project_home_dir, features2dpath):
    """Extract features from CNN csv file for given phase and cvphase, stack based on frametype.
    Manual 2d features concatenated horizontally here (102 added to 256 features to have 358 per frame feature vector).
    Keyword arguments:
    phase -- train, val, trainval or test (which data to use)
    cvphase -- cross validation fold (0 to 4)
    frametype -- adjacent, equalspaced or singleframe (whether to stack frames or not, and how if so)
    Return (for given cross validation fold and train/val/trainval/test phase) lists of all feature vectors, labels, patient IDs and frame numbers within patient."""
    foundcount = 0
    
    trainvalfile = config['featurestrainvalfile']
    testfile = config['featurestestfile']
    filename = config['featurestrainfile']
    valfile = config['featuresvalfile']
    
    
    curfold = 0
    print("CVPHASE:", cvphase, "\ncsv:", filename)
    if(phase == "train"):
        curfile = filename
        if(cvphase == 0):
            curfold = 3
        elif(cvphase == 1):
            curfold = 4
        else:
            curfold = cvphase - 2
    elif(phase == "val"):
        curfile = valfile
        if(cvphase == 4):
            curfold = 0
        else:
            curfold = cvphase + 1
    elif(phase == "test"):
        curfile = testfile
        curfold = cvphase
    elif(phase == "trainval"):
        curfile = trainvalfile
        if(cvphase == 0):
            curfold = 3
        elif(cvphase == 1):
            curfold = 4
        else:
            curfold = cvphase - 2
    
    featurelength = 256+102
        
    cur_all_pat_inds = []
    cur_all_patients = []
    cur_all_labels = []
    cur_all_probs = np.zeros((0,featurelength))
    
    #col 0 = framenum, col 1 = patient id, col 2 = label, rest = probabilities
    rowcount = 0
    with open(curfile, newline='') as infh:
        print("opened csv for cnn features")
        reader = csv.reader(infh)
        rowcount = 0
        for row in reader:
            if(row[3] == ''):
                print("ERROR", row[2])
                continue
            cur_all_pat_inds.append(row[0])
            cur_all_labels.append(row[2])
            cur_all_patients.append(row[1])
            probs = np.array(row[3:], dtype=float)
            probs = np.append(probs, np.zeros((102)))
            
            if(rowcount % 500 == 0):
                print("row:", rowcount)
            
            cur_all_probs = np.vstack((cur_all_probs, probs))
            rowcount += 1
        print("rowcount", rowcount)
            
    print("length of probs, labels, ids", len(cur_all_probs), len(cur_all_labels), len(cur_all_patients))
    
    #get number of images per patient
    distinct_patient_ids = []
    distinct_num_pats = []
    
    #SORT IN ORDER OF FRAME NUM NOW, BEFORE PADDING!
    cur_all_pat_inds = np.array(cur_all_pat_inds).astype(np.float64)
    print("type of cur_all_pat_inds", type(cur_all_pat_inds))
    templist = list(zip(cur_all_patients, cur_all_pat_inds, cur_all_probs, cur_all_labels))
    sortedlist = sorted(templist, key=operator.itemgetter(0, 1))
    cur_all_patients, cur_all_pat_inds, cur_all_probs, cur_all_labels = zip(*sortedlist)
    print("\n\n\n should be sorted (framenums, patient id, label):", cur_all_pat_inds[0:10], cur_all_patients[0:10], cur_all_labels[0:10])
    
    print("\n\nLOADING", phase,"PHASE")
    maxpat = ""
    maxpatcount = 0
    curpatcount = 0
    
    for patt in cur_all_patients:
        if (patt not in distinct_patient_ids):
            curpatcount = 0
            distinct_patient_ids.append(patt)
            for xz in range(len(cur_all_patients)):                
                if(cur_all_patients[xz] == patt):
                    curpatcount += 1
                    if(curpatcount >= maxpatcount):
                        maxpat = patt
                        maxpatcount = curpatcount
                        
            distinct_num_pats.append(curpatcount)
            print("patient", patt, ": CNN extracted # of frames =", curpatcount)
    
    print("cur all probs shape", np.shape(cur_all_probs))
    print("num patients:", len(distinct_patient_ids), " total length:", len(cur_all_patients))
    

    #load manual 2d features
    cur_all_concats, concatpats, concatlabs, concatframenums = process2dfeatures(False, cvphase, features2dpath)

     #loop through columns and normalize
    for colnum in range(len(cur_all_concats[0])):
        cur_all_concats[:, colnum] = normalize(cur_all_concats[:, colnum])

    #need to now add to probs the next two rows and then average before the for loop
    #THIS IS FOR ADJACENT FRAMES!
    print("from concat features: pats", concatpats[int(len(concatpats)/3)])

    for looppats in range(len(cur_all_patients)): #every single index of each individual feature vector instance. should be thousands
        if(looppats % 1000 == 0):
            print("looping through all feature vectors, index", looppats)
        thecurpat = cur_all_patients[looppats]
        thecurframenum = cur_all_pat_inds[looppats]

        patientind = concatpats.index(thecurpat) #find first instance of this patient in concat patient list
        lastpatientind = len(concatpats) - 1 - concatpats[::-1].index(thecurpat) #find last instance of this patient in patient list

        avgfeatures = np.zeros((3,102))
        found = False
        for gg in range(patientind, lastpatientind):
            avgfeatures = np.zeros((3,102))

            if(int(cur_all_pat_inds[looppats]) == int(concatframenums[gg])): #corresponding first frame #
                if(gg>=(len(cur_all_concats)-2)):
                    print("over limit; gg, features", gg, "len avg features and concats", len(avgfeatures), len(cur_all_concats[gg]))
                    gg-=2
                foundcount += 1
                found = True
                avgfeatures[0] = cur_all_concats[gg]#1st row of concat probs
                avgfeatures[1] = cur_all_concats[gg+1]#add 2nd row of concat probs
                avgfeatures[2] = cur_all_concats[gg+2]#add 3rd row of concat probs

                avgfeatures = np.average(avgfeatures, axis=0)

                cur_all_probs[looppats][256:] = avgfeatures

        patienttot = len(cur_all_patients) - 1 - cur_all_patients[::-1].index(thecurpat) - cur_all_patients.index(thecurpat)
        if(not found):
            print("framenum notmatching", thecurpat, "cnn framenum:", cur_all_pat_inds[looppats], "vs.", concatframenums[patientind], "to", concatframenums[lastpatientind])#:lastpatientind])

        """#below is for appending first frame's features, not average of 3 stacked frames
        for gg in range(patientind, lastpatientind):
            if(cur_all_pat_inds[gg] == curpatindexfull[numberind:]):
                cur_all_probs[gg][256:] = probs #np.vstack((cur_all_probs, probs))#np.stack((cur_all_probs, probs))
                #print("ERROR SHAPE IS WRONG!!! in row", rowcount, len(cur_all_probs[gg][256:]), "vs. len probs", len(probs))
        """
    print("\n\nFOUND total of", foundcount,"corresponding 2d features vs. total of", len(cur_all_probs), "frames")

    tempp = list(zip(distinct_num_pats, distinct_patient_ids)) 
    tempp = sorted(tempp) 
    distinct_num_pats, distinct_patient_ids = zip(*tempp)
    
    print(phase, "PATIENTS:", distinct_patient_ids)
    print("cvphase", cvphase,"numpatients input:", len(distinct_patient_ids))
    
    if(phase == "train" or phase == "trainval"):
        config = transfconfigwrite('numpatstrain', len(distinct_patient_ids))
    elif(phase == "val"):
        config = transfconfigwrite('numpatsval', len(distinct_patient_ids))
    elif(phase == "test"):
        config = transfconfigwrite('numpatstest', len(distinct_patient_ids))
    
    curpatprobs = np.zeros((0,featurelength))
    curpatlabs = []
    curpatids = []
    curpatframenums = []
    
    allfeatures = np.zeros((0, featurelength))
    alllabels = []
    allids = []
    allframenums = []
    
    #ADD INDIVIDUAL PATIENT VECTORS TO ALLFEATURES VECTOR
    #pad and stack
    counttt = 0
    seq_len = 36

    for patind in range(len(distinct_patient_ids)):
        curpatprobs = np.zeros((0,featurelength))
        curpatlabs = []
        curpatids = []
        curpatframenums = []
        
        pat = distinct_patient_ids[patind]
        for p in range(len(cur_all_patients)):
            if (cur_all_patients[p] == pat):
                curprobs = cur_all_probs[p]
                curprobs = curprobs.reshape((1,featurelength))
                curlabels = cur_all_labels[p]
                curids = cur_all_patients[p]
                curframenums = cur_all_pat_inds[p]
                
                curpatprobs = np.concatenate((curpatprobs, curprobs), axis=0)
                curpatlabs = np.append(curpatlabs, curlabels)
                curpatids = np.append(curpatids, curids)
                curpatframenums = np.append(curpatframenums, curframenums)
        
        
        numimgs = len(curpatprobs)
        goodlenprobs = np.zeros((seq_len,featurelength))
        #PAD NOW
        if (numimgs > seq_len):
            num_seqs = 1
            lencurseq = 36
            while((numimgs / num_seqs) > seq_len):
                num_seqs += 1
            
            lencurseq = numimgs / num_seqs
            
            for g in range(num_seqs):
                #split into groups of less than 36 vectors of length 256
                start = int(g*lencurseq)
                end = int(((g+1)*lencurseq))
                if(g<(num_seqs)-1):
                    currentseq = curpatprobs[start:end]
                    currentlabs = curpatlabs[start:end]
                    currentids = curpatids[start:end]
                    currentframes = curpatframenums[start:end]
                else:
                    currentseq = curpatprobs[start:]
                    currentlabs = curpatlabs[start:]
                    currentids = curpatids[start:]
                    currentframes = curpatframenums[start:]
                    
                #pad each to 36
                currentseq = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(currentseq), torch.from_numpy(goodlenprobs)], batch_first=True, padding_value=0.0)
                currentseq = np.array(currentseq[0])
                while(len(currentlabs)<seq_len):
                    currentlabs = np.append(currentlabs, currentlabs[0]) #or just '0'?
                    currentids = np.append(currentids, currentids[0])
                    currentframes = np.append(currentframes, 0)
                if((g==0) and (patind == 0)):
                    print("shape of padded sequence, labels, ids:", np.shape(currentseq), np.shape(currentlabs), np.shape(currentids), np.shape(currentframes))
                
                #add 36 at a time from curpat probs to allfeatures
                allfeatures = np.concatenate((allfeatures, currentseq), axis=0)
                alllabels = np.append(alllabels, currentlabs)
                allids = np.append(allids, currentids)
                allframenums = np.append(allframenums, currentframes)
            
        elif (numimgs < seq_len):
            #just pad to 36
            diff = seq_len-len(curpatprobs)
            while(len(curpatlabs)<seq_len):
                curpatlabs = np.append(curpatlabs, curpatlabs[0])
                curpatids = np.append(curpatids, curpatids[0])
                curpatframenums = np.append(curpatframenums, curpatframenums[0])#0)

            currentseq = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(curpatprobs), torch.from_numpy(goodlenprobs)], batch_first=True, padding_value=0.0)
            currentseq = np.array(currentseq[0])
            
            #add the 36 from curpatprobs to allfeatures
            allfeatures = np.concatenate((allfeatures, currentseq), axis=0)
            alllabels = np.append(alllabels, curpatlabs)
            allids = np.append(allids, curpatids)
            allframenums = np.append(allframenums, curpatframenums)

        else:
            allfeatures = np.concatenate((allfeatures, curpatprobs), axis=0)
            alllabels = np.append(alllabels, curpatlabs)
            allids = np.append(allids, curpatids)
            allframenums = np.append(allframenums, curpatframenums)
        
        counttt+= 1
        
        if(phase == "train" or phase == "trainval"):
            config = transfconfigwrite('numpatstrain',  int(len(allfeatures) / 36))
        elif(phase == "val"):
            config = transfconfigwrite('numpatsval',  int(len(allfeatures) / 36))
        elif(phase == "test"):
            config = transfconfigwrite('numpatstest',  int(len(allfeatures) / 36))
        
    print("example features:\n", allfeatures[0][0], allfeatures[0][256], allfeatures[0][300])
    print("\nFINAL features, labels, ids shapes", np.shape(allfeatures), np.shape(alllabels), np.shape(allids), np.shape(allframenums))
    return(allfeatures, alllabels, allids, allframenums) #return feature vectors and labels and patient ids for transformer



def load_csv_padded_transformer_vertical_concat(phase, cvphase, frametype, project_home_dir, features2dpath):
    """Extract features from CNN csv file for given phase and cvphase, stack based on frametype.
    Manual 2d features concatenated vertically here (16 frame's concatenated vectors of 102 padded to 256 features each and appended to 18 other feature vectors per patient to get 36 feature vectors of 256).
    Keyword arguments:
    phase -- train, val, trainval or test (which data to use)
    cvphase -- cross validation fold (0 to 4)
    frametype -- adjacent, equalspaced or singleframe (whether to stack frames or not, and how if so)
    Return (for given cross validation fold and train/val/trainval/test phase) lists of all feature vectors, labels, patient IDs and frame numbers within patient."""
    
    
    trainvalfile = config['featurestrainvalfile']
    testfile = config['featurestestfile']
    filename = config['featurestrainfile']
    valfile = config['featuresvalfile']
    
    print("CVPHASE:", cvphase, "\ncsv:", filename)
    curfold = 0
    if(phase == "train"):
        curfile = filename
        if(cvphase == 0):
            curfold = 3
        elif(cvphase == 1):
            curfold = 4
        else:
            curfold = cvphase - 2
    elif(phase == "val"):
        curfile = valfile
        if(cvphase == 4):
            curfold = 0
        else:
            curfold = cvphase + 1
    elif(phase == "test"):
        curfile = testfile
        curfold = cvphase
    elif(phase == "trainval"):
        curfile = trainvalfile
        if(cvphase == 0):
            curfold = 3
        elif(cvphase == 1):
            curfold = 4
        else:
            curfold = cvphase - 2

    featurelength = 256
    
    cur_all_pat_inds = [] #framenum
    cur_all_patients = []
    cur_all_labels = []
    cur_all_probs = np.zeros((0,featurelength))
    
    rowcount = 0
    with open(curfile, newline='') as infh:
        print("opened csv for cnn features")
        reader = csv.reader(infh)
        rowcount = 0
        for row in reader:
            if(row[3] == ''):
                print("ERROR", row[2])
                continue
            cur_all_pat_inds.append(row[0])
            cur_all_labels.append(row[2])
            cur_all_patients.append(row[1])
            
            probs = np.array(row[3:], dtype=float)

            if(rowcount % 500 == 0):
                print("row:", rowcount)
            #try:
            cur_all_probs = np.vstack((cur_all_probs, probs))
            rowcount += 1
        print("rowcount", rowcount)
            
    print("length of probs, labels, ids", len(cur_all_probs), len(cur_all_labels), len(cur_all_patients))
    
    
    #get number of images per patient
    distinct_patient_ids = []
    distinct_num_pats = []

    print("\n\nLOADING", phase, "PHASE")
    maxpat = ""
    maxpatcount = 0
    curpatcount = 0
    for patt in cur_all_patients:
        if (patt not in distinct_patient_ids):
            curpatcount = 0
            distinct_patient_ids.append(patt)
            for xz in range(len(cur_all_patients)):
                if(cur_all_patients[xz] == patt):
                    curpatcount += 1
                    if(curpatcount >= maxpatcount):
                        maxpat = patt
                        maxpatcount = curpatcount
            distinct_num_pats.append(curpatcount)
    
    print("cur all probs shape", np.shape(cur_all_probs), "concats", np.shape(cur_all_concats))
    print("num patients:", len(distinct_patient_ids), " total length:", len(cur_all_patients))
    
    #manual 2d features to concatenate:
    cur_all_concats, concatpats, concatlabs, concatframenums = process2dfeatures(True, cvphase, features2dpath)
    
    #SORT IN ORDER OF FRAME NUM NOW, BEFORE PADDING!
    cur_all_pat_inds = np.array(cur_all_pat_inds).astype(np.float64)
    print("type of cur_all_pat_inds", type(cur_all_pat_inds))
    templist = list(zip(cur_all_patients, cur_all_pat_inds, cur_all_probs, cur_all_labels))
    sortedlist = sorted(templist, key=operator.itemgetter(0, 1))
    cur_all_patients, cur_all_pat_inds, cur_all_probs, cur_all_labels = zip(*sortedlist)
    print("\n\n\n should be sorted (framenums, patient id, label):", cur_all_pat_inds[0:10], cur_all_patients[0:10], cur_all_labels[0:10])
    
    tempp = list(zip(distinct_num_pats, distinct_patient_ids)) 
    tempp = sorted(tempp) 
    distinct_num_pats, distinct_patient_ids = zip(*tempp)
    
    print(phase, "PATIENTS:", distinct_patient_ids) #all sorted by patient id by now
    if(phase == "train" or phase == "trainval"):
        config = transfconfigwrite('numpatstrain', len(distinct_patient_ids))
    elif(phase == "val"):
        config = transfconfigwrite('numpatsval', len(distinct_patient_ids))
    elif(phase == "test"):
        config = transfconfigwrite('numpatstest', len(distinct_patient_ids))
    
    curpatprobs = np.zeros((0,featurelength))
    curpatlabs = []
    curpatids = []
    curpatframenums = []
    
    allfeatures = np.zeros((0, featurelength))
    alllabels = []
    allids = []
    allframenums = []
    
    #ADD INDIVIDUAL PATIENT VECTORS TO ALLFEATURES VECTOR
    #pad and stack
    counttt = 0
    seq_len = 18 #18 from CNN and then their corresponding 18 from manual features csv file
    
    #append all features from patient into single vector, curpatprobs
    for patind in range(len(distinct_patient_ids)):
        curpatprobs = np.zeros((0,featurelength))
        curpatprobsconcat = np.zeros((0,featurelength))
        curpatlabs = []
        curpatids = []
        curpatframenums = []
        
        pat = distinct_patient_ids[patind]
        for p in range(len(cur_all_patients)):
            if (cur_all_patients[p] == pat):
                curprobs = cur_all_probs[p]
                curprobs = curprobs.reshape((1,featurelength))
                curlabels = cur_all_labels[p]
                curids = cur_all_patients[p]
                curframenums = cur_all_pat_inds[p]
                
                curpatprobs = np.concatenate((curpatprobs, curprobs), axis=0)
                curpatlabs = np.append(curpatlabs, curlabels)
                curpatids = np.append(curpatids, curids)
                curpatframenums = np.append(curpatframenums, curframenums)

        #need to now add to probs the next two rows and then average before the for loop
        patientind = concatpats.index(pat) #find first instance of this patient in concat patient list
        lastpatientind = len(concatpats) - 1 - concatpats[::-1].index(curpatids[0]) #find last instance of this patient in patient list
        if(patind % 100 == 0):
            print("looping through all patients, index", patind, "and curframenums:", curpatframenums, "vs.", concatframenums[patientind:lastpatientind])

        for cnnframe in range(len(curpatframenums)):
            for gg in range(patientind, lastpatientind):
                avgfeatures = np.zeros((3,featurelength))
                if(int(curpatframenums[cnnframe]) == int(concatframenums[gg])): #corresponding first frame #
                    if(gg>=(len(cur_all_concats)-2)):
                        print("over limit; gg, features", gg, "len avg features and concats", len(avgfeatures), len(cur_all_concats[gg]))
                        gg-=2
                    avgfeatures[0] = cur_all_concats[gg]#1st row of concat probs
                    avgfeatures[1] = cur_all_concats[gg+1]#add 2nd row of concat probs
                    avgfeatures[2] = cur_all_concats[gg+2]#add 3rd row of concat probs
                    avgfeatures = np.average(avgfeatures, axis=0)
                    curpatprobsconcat = np.concatenate((curpatprobsconcat, np.reshape(avgfeatures, (1, len(avgfeatures)))), axis=0)#avgfeatures), axis=0)
        if(len(curpatprobs) != len(curpatprobsconcat)):
            print("different pat probs (", len(curpatprobs),") and concat # feature vectors (", len(curpatprobsconcat),")", "curpatframenums", curpatframenums, "vs concat framenums",concatframenums[patientind:lastpatientind])
        
        
        numimgs = len(curpatprobs)
        goodlenprobs = np.zeros((seq_len,featurelength))
        #PAD NOW
        if (numimgs > seq_len):
            num_seqs = 1
            lencurseq = seq_len
            while((numimgs / num_seqs) > seq_len):
                num_seqs += 1
            
            lencurseq = numimgs / num_seqs
            
            for g in range(num_seqs):
                #split into groups of less than 36 vectors of length 256
                start = int(g*lencurseq)
                end = int(((g+1)*lencurseq))
                if(g<(num_seqs)-1):
                    currentseq = curpatprobs[start:end]
                    currentlabs = curpatlabs[start:end]
                    currentids = curpatids[start:end]
                    currentframes = curpatframenums[start:end]
                    currentconcatseq = curpatprobsconcat[start:end] #new
                    
                else:
                    currentseq = curpatprobs[start:]
                    currentlabs = curpatlabs[start:]
                    currentids = curpatids[start:]
                    currentframes = curpatframenums[start:]
                    currentconcatseq = curpatprobsconcat[start:]
                    
                #pad each to 36
                currentseq = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(currentseq), torch.from_numpy(goodlenprobs)], batch_first=True, padding_value=0.0)
                currentseq = np.array(currentseq[0])
                currentconcatseq = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(currentconcatseq), torch.from_numpy(goodlenprobs)], batch_first=True, padding_value=0.0)
                currentconcatseq = np.array(currentconcatseq[0])
                
                while(len(currentlabs) < seq_len):
                    currentlabs = np.append(currentlabs, currentlabs[0]) #or just '0'?
                    currentids = np.append(currentids, currentids[0])
                    currentframes = np.append(currentframes, 0)
                if((g == 0) and (patind == 0)):
                    print("shape of padded sequence, labels, ids:", np.shape(currentseq), np.shape(currentlabs), np.shape(currentids), np.shape(currentframes), "concats", np.shape(currentconcatseq))
                    print(currentlabs[30:36], currentids[16:22], currentids[30:36], currentframes[30:36], currentframes[16:22])

                #add 36 at a time from curpat probs to allfeatures
                allfeatures = np.concatenate((allfeatures, currentseq), axis=0)
                alllabels = np.append(alllabels, currentlabs)
                allids = np.append(allids, currentids)
                allframenums = np.append(allframenums, currentframes)
                
                #now add concats and labels etc. once again (to make 36)
                allfeatures = np.concatenate((allfeatures, currentconcatseq), axis=0)
                alllabels = np.append(alllabels, currentlabs)
                allids = np.append(allids, currentids)
                allframenums = np.append(allframenums, currentframes)
                
            
        elif (numimgs < seq_len):
            #just pad to 36 
            diff = seq_len-len(curpatprobs)
            while(len(curpatlabs)<seq_len):
                curpatlabs = np.append(curpatlabs, curpatlabs[0])
                curpatids = np.append(curpatids, curpatids[0])
                curpatframenums = np.append(curpatframenums, curpatframenums[0])#0)
            
            currentseq = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(curpatprobs), torch.from_numpy(goodlenprobs)], batch_first=True, padding_value=0.0)
            currentseq = np.array(currentseq[0])
            currentconcatseq = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(curpatprobsconcat), torch.from_numpy(goodlenprobs)], batch_first=True, padding_value=0.0)
            currentconcatseq = np.array(currentconcatseq[0])
            
            #add the 36 from curpatprobs to allfeatures
            allfeatures = np.concatenate((allfeatures, currentseq), axis=0)
            alllabels = np.append(alllabels, curpatlabs)
            allids = np.append(allids, curpatids)
            allframenums = np.append(allframenums, curpatframenums)
            
            #now add concats and labels etc. once again (to make 36 or 72)
            allfeatures = np.concatenate((allfeatures, currentconcatseq), axis=0)
            alllabels = np.append(alllabels, curpatlabs)
            allids = np.append(allids, curpatids)
            allframenums = np.append(allframenums, curpatframenums)
            if(len(allfeatures) != len(alllabels)):
                print("in <36 patient", pat,"wrong lengths:", "\nFINAL features, labels, ids shapes", np.shape(currentseq), np.shape(currentconcatseq), np.shape(curpatlabs))
                
        else:
            allfeatures = np.concatenate((allfeatures, curpatprobs), axis=0)
            alllabels = np.append(alllabels, curpatlabs)
            allids = np.append(allids, curpatids)
            allframenums = np.append(allframenums, curpatframenums)
            
            #now add concats and labels etc. once again (to make 36)
            allfeatures = np.concatenate((allfeatures, curpatprobsconcat), axis=0)
            alllabels = np.append(alllabels, curpatlabs)
            allids = np.append(allids, curpatids)
            allframenums = np.append(allframenums, curpatframenums)
            
            if(len(allfeatures) != len(alllabels)):
                print("in else (len 36) patient", pat,"wrong lengths:", np.shape(curpatprobs), np.shape(curpatprobsconcat), np.shape(curpatlabs))
            print("\current features, labels, ids shapes", np.shape(allfeatures), np.shape(alllabels), np.shape(allids), np.shape(allframenums))
        
        counttt+= 1
        
        if(phase == "train" or phase == "trainval"):
            config = transfconfigwrite('numpatstrain', int(len(allfeatures) / (2*seq_len)))
        elif(phase == "val"):
            config = transfconfigwrite('numpatsval', int(len(allfeatures) / (2*seq_len)))
        elif(phase == "test"):
            config = transfconfigwrite('numpatstest', int(len(allfeatures) / (2*seq_len)))
        
    print("example features:\n", allfeatures[0][0], allfeatures[0][100])
    print("\nFINAL features, labels, ids shapes", np.shape(allfeatures), np.shape(alllabels), np.shape(allids), np.shape(allframenums))
    return(allfeatures, alllabels, allids, allframenums) #return feature vectors and labels and patient ids for transformer





class DatasetPaddedTransformer(data.Dataset):
    def __init__(self, phase, cvphase, concat_features, vertconcat, frametype, project_home_dir):
        """Data loader for Transformer Model."""
        super(DatasetPaddedTransformer, self).__init__()
        
        features2dpath = '/your/2d/features/path.csv'
        
        self.phase = phase
        if(concat_features): #horizontal concat or vertical concatenated 2d features
            if(vertconcat):
                self.all_frames, self.all_labels, self.all_annot_ids, self.all_frame_nums = load_csv_padded_transformer_vertical_concat(phase, cvphase, frametype, project_home_dir, features2dpath)
            else:
                self.all_frames, self.all_labels, self.all_annot_ids, self.all_frame_nums = load_csv_padded_transformer_horiz_concat(phase, cvphase, frametype, project_home_dir, features2dpath)
        else: #no concatenated 2d features
            self.all_frames, self.all_labels, self.all_annot_ids, self.all_frame_nums = load_csv_padded_transformer(phase, cvphase, frametype, project_home_dir)
                
        self.seqlength = 36
        self.start_indices = []
        numiters = len(self.all_frames) / self.seqlength
        print("numiters", numiters)
        for x in range(int(numiters)):
            self.start_indices.append(x*self.seqlength)
        print("done reading in images and labels for", phase, "!!!\n\n")
        print("start indices (should be multiples of sequence length only):", self.start_indices)
        print("all frames:", len(self.all_frames), np.shape(self.all_frames))
    
    
    #getitem is called 'batch_size' number of times in one iteration of the epoch
    def __getitem__(self, i):
        startind = self.start_indices[i]
        try:
            endind = self.start_indices[i+1] #this is the first of the NEXT patient so do NOT INCLUDE!
        except: #last patient
            endind = len(self.all_labels)
        
        img_frames = self.all_frames[startind:endind] #feature vectors. img_frames should be size (numimgsinpatient, 2)
        annot_ids = self.all_annot_ids[startind:endind] #should be all the same, annot_ids should be size (numimgsinpatient)
        intannot_ids = []
        
        #create labels for images
        labels = torch.LongTensor(endind-startind)
        for l in range(0, endind-startind):
            labels[l] = int(self.all_labels[startind+l]) #should be all the same, labels should be size (numimgsinpatient)
            intannot_ids.append(int(annot_ids[l][:-1])) #make into integer
        
        inputs = torch.from_numpy(img_frames).float()
        
        intannot_ids = np.array(intannot_ids)
        intannot_ids = torch.from_numpy(intannot_ids)
        
        framenums = self.all_frame_nums[startind:endind] #frame nums, should be in order

        for i in range(len(framenums)):
            framenums[i] = float(framenums[i])
        framenums = np.array(framenums, dtype=np.float32)
        framenums = torch.from_numpy(framenums).float()
        
        return {'input': inputs, 'label': labels, 'annot_id': intannot_ids, 'framenum': framenums}

    def __len__(self):
        return len(self.all_frames)


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]