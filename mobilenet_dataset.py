print("\nTHYROID DATASET\n")
import pandas as pd
from PIL import Image
import os
import os.path

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


import mobilenet_preprocess

def load_datasets_new(project_home_dir, labelpath, phase, cv_phase, allimgs, frametype):
    """Load images and labels for given phase and cvphase, stack 3 different frames based on frametype (adjacent, equally spaced).
    Keyword arguments:
    phase -- train, val, trainval or test (which data to use)
    cv_phase -- cross validation fold (0 to 4)
    frametype -- adjacent, equalspaced (how to stack frames)
    Return (for given cross validation fold and train/val/trainval/test phase) lists of all images (stacked), labels, patient IDs and frame numbers within patient."""

    print("passed in list allimgs:", np.shape(allimgs))
    
    colnames = ['Labels for each frame', 'Annot_IDs', 'size_A', 'size_B', 'size_C', 'location_r_l_', 'study_dttm', 'age', 'sex', 'final_diagnoses', 'ePAD ID', 'foldNum']
    label_data = pd.read_csv(labelpath, names=colnames)
    
    annot_ids = label_data.Annot_IDs.tolist() #list of annotation ids from csv file
    labels = label_data.final_diagnoses.tolist() #list of labels from csv file
    
    foldNums = label_data.foldNum.tolist() #list of what folder for train test split
    
    annot_ids.pop(0)
    labels.pop(0)
    foldNums.pop(0)
    
    correct_order_labels = []
    
    for i in range(len(allimgs)):
        correct_order_labels.append(labels[i])
    
    print('Num Images: {}\n Labels: {}\n'.format(len(allimgs), len(correct_order_labels), len(annot_ids)))
    cur_imgs = []
    cur_labels = []
    cur_annot_ids = []

    test_folder = cv_phase #0, 1, 2, 3, 4
    if(cv_phase < 4):
        val_folder = cv_phase+1
    elif(cv_phase == 4):
        val_folder = 0
    
    print("CROSS VALIDATION PHASE:", test_folder)
    #split by foldnum group
    for g in range(len(allimgs)):
        fnum = int(foldNums[g])
        
        if (phase == "train"):
            if not (fnum == test_folder or fnum == val_folder):
                cur_imgs.append(allimgs[g])
                cur_labels.append(correct_order_labels[g])
                cur_annot_ids.append(annot_ids[g])
        elif (phase == "val"):
            if (fnum == val_folder):
                cur_imgs.append(allimgs[g])
                cur_labels.append(correct_order_labels[g])
                cur_annot_ids.append(annot_ids[g])
        elif (phase == "trainval"):
            if not (fnum == test_folder): #train and val images
                cur_imgs.append(allimgs[g])
                cur_labels.append(correct_order_labels[g])
                cur_annot_ids.append(annot_ids[g])
        #else: #test phase
        elif (phase == "test"):
            if (fnum == test_folder):
                cur_imgs.append(allimgs[g])
                cur_labels.append(correct_order_labels[g])
                cur_annot_ids.append(annot_ids[g])
        
    
    #label frame number of each image
    cur_frame_num = []
    distinct_patient_ids = []
    patientframenum = 0

    #if 3 images in a row from same patient, stack them (instead of rgb channels)
    for ind in range(len(cur_imgs)):
        if(not (cur_annot_ids[ind] in distinct_patient_ids)):
            distinct_patient_ids.append(cur_annot_ids[ind])
            patientframenum = 1
        cur_frame_num.append(patientframenum) #add index of first image in current frame stack within patient to list for order
        patientframenum += 1
    print(len(cur_imgs), len(cur_labels), len(cur_annot_ids), len(cur_frame_num))

    
    #stack:
    cur_imgs_stack = []
    cur_labels_stack = []
    cur_annot_ids_stack = []
    cur_frame_num_stack = []
    
    distinct_patient_ids = []
    dist = 10
    t = 0
    
    print("about to stack!")
    
    if(frametype == "adjacent"):
        #ADJACENT frame stacking method
        #if 3 images in a row (adjacent) from same patient, stack them (instead of rgb channels)
        while (t < len(cur_imgs)-2):
            if(cur_annot_ids[t] not in distinct_patient_ids):
                distinct_patient_ids.append(cur_annot_ids[t])
            if(cur_annot_ids[t] == cur_annot_ids[t+1] and cur_annot_ids[t] == cur_annot_ids[t+2]):
                img = np.stack((cur_imgs[t], cur_imgs[t+1], cur_imgs[t+2]))
                cur_imgs_stack.append(img)
                cur_labels_stack.append(cur_labels[t])
                cur_annot_ids_stack.append(cur_annot_ids[t])
                cur_frame_num_stack.append(cur_frame_num[t]) #add index of first image in current frame stack within patient to list for order
                if(cur_labels[t] != cur_labels[t+1] or cur_labels[t] != cur_labels[t+2]):
                    print("inconsistent labels in train group of 3 images!")
            t += 3 #every option needs t + 1 to go to next frame   
   
    else:
        #EQUAL SPACED frame stacking method
        while (t < len(cur_imgs) - (2*dist)):
            if(cur_annot_ids[t] not in distinct_patient_ids):
                annot_id = cur_annot_ids[t]
                #last index of this patient id in the list of ids
                last_id = len(cur_annot_ids) - cur_annot_ids[::-1].index(annot_id) - 1
                #print("num in this patient", last_id - t)
                dist = (last_id - t) // 3
                distinct_patient_ids.append(cur_annot_ids[t])
                print("PATIENT", cur_annot_ids[t], "num frames:", last_id-t)

            #if 3 images equally spaced from same patient, stack them (instead of rgb channels)
            if(t <= (last_id - (2*dist))):
                if(cur_annot_ids[t] == cur_annot_ids[t+dist] and cur_annot_ids[t] == cur_annot_ids[t+(2*dist)]):
                    img = np.stack((cur_imgs[t], cur_imgs[t+dist], cur_imgs[t+(2*dist)]))
                    cur_imgs_stack.append(img)
                    cur_labels_stack.append(cur_labels[t])
                    cur_annot_ids_stack.append(cur_annot_ids[t])
                    cur_frame_num_stack.append(cur_frame_num[t]) #add index of first image in current frame stack within patient to list for order

                    if(cur_labels[t] != cur_labels[t+dist] or cur_labels[t] != cur_labels[t+(2*dist)]):
                        print("inconsistent labels in train group of 3 images!")
                        print("t:", t, "dist:", dist, "last id", last_id, "cur labels:", cur_labels[t], cur_annot_ids[t], cur_labels[t+(2*dist)], cur_annot_ids[t+(2*dist)])
            else: #end of patient?
                g = 0
                if(cur_annot_ids[t] != cur_annot_ids[t + (2*dist) + 1]):
                    t += (2*dist) #go to next patient!
                else:
                    print("not at end of patient?")
                    while(cur_annot_ids[t] == cur_annot_ids[t + (2*dist) + 1 + g]):
                        g += 1
                    print("final difference:", g)
                    t += (2*dist) + g
            t += 1 #every option needs t + 1 to go to next frame
            
    print("\n\nnum patients:", len(distinct_patient_ids))
    print("done stacking!")
    #shuffle all lists with same order for TRAIN ONLY
    if (phase == 'train' or phase == 'trainval'):# or phase == 'test'):
        temp = list(zip(cur_imgs_stack, cur_labels_stack, cur_annot_ids_stack, cur_frame_num_stack)) 
        random.shuffle(temp) 
        cur_imgs_stack, cur_labels_stack, cur_annot_ids_stack, cur_frame_num_stack = zip(*temp)
    
    if (phase == 'train' or phase == 'trainval'):
        #for class weights (imbalance of classes)
        neg, pos = np.bincount(cur_labels_stack)#intlabels)
        print("0s", neg, "1s", pos)
        total_lbls = neg + pos
        print(total_lbls == len(cur_labels_stack))
        print('Labels:\n Total: {}\n Positive: {} ({:.2f}% of total)\n'.format(total_lbls, pos, 100 * pos / total_lbls))

        #https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        # Scaling by total/2 helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.
        weight_for_0 = (1 / neg)*(total_lbls)/2.0 
        weight_for_1 = (1 / pos)*(total_lbls)/2.0

        #class_weight = {0: weight_for_0, 1: weight_for_1}
        class_weight = [weight_for_0, weight_for_1]
        print('Weight for class 0: {:.2f}\nWeight for class 1: {:.2f}'.format(weight_for_0, weight_for_1))
        samples_weight = []
        samples_weight = np.array([class_weight[int(m)] for m in cur_labels_stack])
        print("in train phase for load_dataset: samples_weight shape =", np.shape(samples_weight))
        
        f=open(project_home_dir + "samplesweight.csv",'w', newline ='\n')
        count = 0
        for s in zip(samples_weight):
            count += 1
            f.write(str(s[0])+",")
        f.close()
        
    #done preprocessing!
    print("len", phase, "=", len(cur_imgs))
    print("len", phase, "(stacked)", len(cur_imgs_stack), len(cur_labels_stack), len(cur_annot_ids_stack), len(cur_frame_num_stack), np.shape(cur_imgs_stack))
    return(cur_imgs_stack, cur_labels_stack, cur_annot_ids_stack, cur_frame_num_stack) #return train/test imgs and labels and frame index #s



#only max area frame +- 1 frame from each patient
def load_datasets_single_frame(project_home_dir, labelpath, phase, cv_phase, allimgs, largestpatinds):
    """Load images and labels for given phase and cvphase, only stack largest frame per patient +- 1 frame.
    Keyword arguments:
    phase -- train, val, trainval or test (which data to use)
    cv_phase -- cross validation fold (0 to 4)
    largest_pat_inds -- list with 1 at index of largest image in each patient (from transform_and_crop_largest function in mobilenet_preprocess); use this image +- 1 image for stacking.
    Return (for given cross validation fold and train/val/trainval/test phase) lists of all images (stacked), labels, patient IDs and frame numbers within patient."""

    print("passed in list allimgs:", np.shape(allimgs))
    
    colnames = ['Labels for each frame', 'Annot_IDs', 'size_A', 'size_B', 'size_C', 'location_r_l_', 'study_dttm', 'age', 'sex', 'final_diagnoses', 'ePAD ID', 'foldNum']
    label_data = pd.read_csv(labelpath, names=colnames)
    
    annot_ids = label_data.Annot_IDs.tolist() #list of annotation ids from csv file
    labels = label_data.final_diagnoses.tolist() #list of labels from csv file
    
    foldNums = label_data.foldNum.tolist() #list of what folder for train test split
    
    annot_ids.pop(0)
    labels.pop(0)
    foldNums.pop(0)
    
    correct_order_labels = []
    
    selannotids = []
    selimgs = []
    selfoldnums = []

    #only use largest in patient
    for i in range(len(allimgs)-1):
        if(largestpatinds[i] == 1 or largestpatinds[i+1] == 1 or largestpatinds[i-1] == 1): #largest, and before and after (3 per patient)
            correct_order_labels.append(labels[i])
            selannotids.append(annot_ids[i])
            selimgs.append(allimgs[i])
            selfoldnums.append(foldNums[i])
    
    print('Num TOTAL Images: {}\n patients: {}\n'.format(len(allimgs), len(annot_ids)))
    print('Num SELECTED (max) Images: {}\n Labels: {}\n Patients: {}\n'.format(len(selimgs), len(correct_order_labels), len(selannotids)))
    
    allimgs = selimgs
    annot_ids = selannotids
    foldNums = selfoldnums
    
    cur_imgs = []
    cur_labels = []
    cur_annot_ids = []

    test_folder = cv_phase #0, 1, 2, 3, 4
    if(cv_phase < 4):
        val_folder = cv_phase+1
    elif(cv_phase == 4):
        val_folder = 0
    
    print("CROSS VALIDATION PHASE:", test_folder)
    #split by foldnum group
    for g in range(len(allimgs)):
        fnum = int(foldNums[g])
        
        if (phase == "train"):
            if not (fnum == test_folder or fnum == val_folder):
                cur_imgs.append(allimgs[g])
                cur_labels.append(correct_order_labels[g])
                cur_annot_ids.append(annot_ids[g])
        elif (phase == "val"):
            if (fnum == val_folder):
                cur_imgs.append(allimgs[g])
                cur_labels.append(correct_order_labels[g])
                cur_annot_ids.append(annot_ids[g])
        elif (phase == "trainval"):
            if not (fnum == test_folder): #train and val images
                cur_imgs.append(allimgs[g])
                cur_labels.append(correct_order_labels[g])
                cur_annot_ids.append(annot_ids[g])
        #else: #test phase
        elif (phase == "test"):
            if (fnum == test_folder):
                cur_imgs.append(allimgs[g])
                cur_labels.append(correct_order_labels[g])
                cur_annot_ids.append(annot_ids[g])
        
    #stack:
    cur_imgs_stack = []
    cur_labels_stack = []
    cur_annot_ids_stack = []
    #label frame number of each image (for future ordering)
    cur_frame_num = []
    
    distinct_patient_ids = []
    t = 0
    patientframenum = 1
    
    print("about to stack!")
    for t in range(len(cur_imgs)):
        if(cur_annot_ids[t] not in distinct_patient_ids):
            annot_id = cur_annot_ids[t]
            distinct_patient_ids.append(cur_annot_ids[t])
            patientframenum = 1
        
        img = np.stack((cur_imgs[t], cur_imgs[t], cur_imgs[t]))
        cur_imgs_stack.append(img)
        cur_labels_stack.append(cur_labels[t])
        cur_annot_ids_stack.append(cur_annot_ids[t])
        cur_frame_num.append(patientframenum) #add index of first image in current frame stack within patient to list for order
        patientframenum += 1

    print("done stacking!")
    #shuffle all lists with same order for TRAIN ONLY
    if (phase == 'train' or phase == 'trainval'):
        temp = list(zip(cur_imgs_stack, cur_labels_stack, cur_annot_ids_stack)) 
        random.shuffle(temp) 
        cur_imgs_stack, cur_labels_stack, cur_annot_ids_stack = zip(*temp)
    
    if (phase == 'train' or phase == 'trainval'):
        #for class weights (imbalance of classes)
        neg, pos = np.bincount(cur_labels_stack)
        print("0s", neg, "1s", pos)
        total_lbls = neg + pos
        print(total_lbls == len(cur_labels_stack))
        print('Labels:\n Total: {}\n Positive: {} ({:.2f}% of total)\n'.format(total_lbls, pos, 100 * pos / total_lbls))
        #https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        # Scaling by total/2 helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.
        weight_for_0 = (1 / neg)*(total_lbls)/2.0 
        weight_for_1 = (1 / pos)*(total_lbls)/2.0

        class_weight = [weight_for_0, weight_for_1]
        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))
        samples_weight = []
        samples_weight = np.array([class_weight[int(m)] for m in cur_labels_stack])
        print("in train phase for load_dataset: samples_weight shape =", np.shape(samples_weight))
        f=open(project_home_dir + "samplesweight.csv",'w', newline ='\n')
        count = 0
        for s in zip(samples_weight):
            count += 1
            f.write(str(s[0])+",")
        f.close()
        
    #done preprocessing!
    print("len", phase, "=", len(cur_imgs))
    print("len", phase, "stacked =", len(cur_imgs_stack))
    print("len", phase, "(stacked)", len(cur_imgs_stack), len(cur_labels_stack), len(cur_annot_ids_stack), np.shape(cur_imgs_stack), len(cur_frame_num))
    return(cur_imgs_stack, cur_labels_stack, cur_annot_ids_stack, cur_frame_num) #return train/test imgs and labels and frame index #s


class DatasetThyroid3StackedNew(data.Dataset):
    def __init__(self, imgpath, maskpath, labelpath, project_home_dir, phase, cvphase, frametype, transform=None):
        """Data loader for CNN Model."""
        super(DatasetThyroid3StackedNew, self).__init__()
        
        h5py.File(imgpath).keys()
        colnames = ['Labels for each frame', 'Annot_IDs', 'size_A', 'size_B', 'size_C', 'location_r_l_', 'study_dttm', 'age', 'sex', 'final_diagnoses', 'ePAD ID', 'foldNum']
        imgs, largestpatinds = mobilenet_preprocess.transform_and_crop_largest(h5py.File(imgpath)['img'], h5py.File(maskpath)['img'], pd.read_csv(labelpath, names=colnames).Annot_IDs.tolist())

        self.phase = phase
        
        print("frametype", frametype)
        if(frametype == "singleframe"):
            self.imgs, self.all_labels, self.all_annot_ids, self.all_frame_nums = load_datasets_single_frame(project_home_dir, labelpath, phase, cvphase, imgs, largestpatinds)
        if(frametype == "adjacent" or frametype == "equalspaced"):
            self.imgs, self.all_labels, self.all_annot_ids, self.all_frame_nums = load_datasets_new(project_home_dir, labelpath, phase, cvphase, imgs, frametype)
        print("done reading in images and labels for", phase, "!!!\n\n")
    
        imgs = []
    
        self.transform = transform
        print("all frames:", len(self.imgs), np.shape(self.imgs))


    
    #getitem is called 'batch_size' number of times in one iteration of the epoch
    def __getitem__(self, i):
        img_frame = self.imgs[i] #3 stacked frames (rgb) from same patient OR same image x3
        annot_id = self.all_annot_ids[i]
        frame_num = self.all_frame_nums[i]
        
        #create label for image
        label = torch.LongTensor(1)
        label[0] = int(self.all_labels[i])

        if(self.transform):
            #make height, width, channels instead of [3,224,224] which is channels, height, width
            img_frame = np.transpose(img_frame, (1,2,0)).astype(np.float32)
            input1 = self.transform(image=img_frame)['image']
        else:
            #doing this in albumentations transform
            input1 = torch.from_numpy(img_frame).float()
            print("no transform")
            
        return {'input': input1, 'label': label, 'annot_id': annot_id, 'frame_num': frame_num}

    def __len__(self):
        return len(self.all_annot_ids)

    
    