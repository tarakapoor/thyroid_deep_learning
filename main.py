# pytorch mobilenet model train
# by tara kapoor

#import torch
import glob

import random
import argparse
import time

#needed?
#import torch
#import torch.nn as nn
#import torchvision
#import torchvision.models as models
#import torchvision.transforms as transforms 

#import torch.optim as optim
#import torch.utils.data as data
#from torch.utils.data import DataLoader
#from sklearn.utils import resample
##

# import sklearn
# from sklearn import metrics
# from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split


# import math
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# print("done with imports!")

#data aug
# from albumentations.pytorch import ToTensorV2
# from albumentations.pytorch import ToTensor


#importing files now

import parsing_file #either do "from __ import functionname" and just use functionname, or "import ___" and then do ___.functionname
parser = parsing_file.create_parser() 
args = parser.parse_args("")
print("Home directory argument in my code:{}".format(args.project_home_dir))

#or 
"""parser = argparse.ArgumentParser(description="Thyroid Project Pytorch Model Code")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes (benign/malignant)")
parser.add_argument("--num_epochs", type=str, default="100", help="how many epochs to train")
parser.add_argument("--best_epoch", type=str, default="99", help="how many epochs to train")
parser.add_argument("--best_thresh", type=float, default=0.0, help="probability threshold from train transformer for benign/malignant prediction")
parser.add_argument("--best_thresh_2tpr", type=float, default=0.0, help="2*tpr weighted probability threshold from train transformer for benign/malignant prediction")
parser.add_argument("--best_thresh_2fpr", type=float, default=0.0, help="2*fpr weighted probability threshold from train transformer for benign/malignant prediction")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--model_dir", type=str, default='_thyroid_weighted_focal_extralinear_3rowstack_trainonly', help="folder to save your trained model")
parser.add_argument("--model_path", type=str, default='', help="actual path to save your trained model")
parser.add_argument("--pretrained_dir", type=str, help="folder to save pretrained model weights for setup_model function")
parser.add_argument("--modeltype", type=str, default='mobilenet', help="type of cnn model running")
parser.add_argument("--phase", type=str, default="train", help="train, val, test, etc")
parser.add_argument("--cvphase", type=int, default=0, help="cross validation fold 0 through 4")
parser.add_argument("--imgpath", type=str, default="/home/jupyter/containers/images/ALL_IMG_Final.hdf5", help="path to img hdf5 file")  #CHANGE if required
parser.add_argument("--maskpath", type=str, default="/home/jupyter/containers/images/ALL_MASK_Final.hdf5", help="path to mask hdf5 file")  #CHANGE if required
parser.add_argument("--labelpath", type=str, default="/home/jupyter/containers/images/frame_lesionId_label_demographics_foldNum.csv", help="path to labels csv file") #CHANGE if required
parser.add_argument("--project_home_dir", type=str, default="/home/jupyter/", help="home project directory") #CHANGE
parser.add_argument("--frametype", type=str, default="adjacent", help="adjacent, equalspaced, singleframe (to feed into model)")
parser.add_argument("--probFunc", type=str, default="Softmax", help="classification function")
parser.add_argument("--weightdecay", type=bool, default=False, help="whether to add weight decay in transformer training")
parser.add_argument("--trainval", type=bool, help="whether in training mode with or without validation data (3/5 or 4/5)")
parser.add_argument("--weighted_sampler", type=bool, default=True)
args = parser.parse_args("")
print(args.phase)"""

#from model_setup import * #has setup_model and save_networks functions
#import mobilenet_preprocess
from mobilenet_dataset import *

from cnn_data_augmentations import * #augmentations are not functions so not sure what this is doing
import train_cnn #train_model (for cnn)
from test_cnn import *
from analyze_model_outputs import * #analyze_test_outputs, plot_test_stats, calc_test_stats, bootstrap_auc

def main():
    #test out dataset functions (don't need if calling in dataloader)
    h5py.File(args.imgpath).keys()
    colnames = ['Labels for each frame', 'Annot_IDs', 'size_A', 'size_B', 'size_C', 'location_r_l_', 'study_dttm', 'age', 'sex', 'final_diagnoses', 'ePAD ID', 'foldNum']
    
    #not needed; done in dataset formation
    #imgs, largestpatinds = transform_and_crop_largest(h5py.File(args.imgpath)['img'], h5py.File(args.maskpath)['img'], pd.read_csv(args.labelpath, names=colnames).Annot_IDs.tolist())

    print("back in main, what is args.cvphase?", args.cvphase)
    args.trainval = False #use 3/5 of data for train, with val
    args.cvphase = 0

    #first round training
    args.trainval = False
    #args.best_epoch = train_cnn.train_model(args)

    #second round training (4/5 of data)
    args.trainval = True #use 4/5 of data for train, no val
    print(args.best_epoch)
    #realtrain = train_model(args) #ignore this returned epoch
    print("\n\n4/5 data model done training!!!")

    #load test data
    epochs = [72, 60, 88, 94, 61]
    args.best_epoch = epochs[args.cvphase]
    print(args.best_epoch)
    print("TEST: cv phase", args.cvphase)
    test_model(args, args.best_epoch)

    #analyze test outputs
    testsaveallfile = "cnn_test_all_outs" + str(args.cvphase) + ".csv"
    analyze_test_outputs(testsaveallfile, "cnn")


    
if __name__ == "__main__":
    # execute only if run as a script
    main()


# print("CV PHASE", args.cvphase)
# test_all_labels = []
# test_all_patients = []
# test_all_probs_ones = []
# rowcount = 0
# with open(testsaveallfile, newline='') as infh:
#     print("opened csv for test outputs")
#     reader = csv.reader(infh)
#     rowcount = 0
#     for row in reader:
#         test_all_patients.append(row[0])
#         test_all_labels.append(row[1])
#         test_all_probs_ones.append(row[2])
#         if(rowcount % 500 == 0):
#             print("row:", rowcount)
#         rowcount += 1
#     print("rowcount", rowcount)
# test_all_patients.pop(0)
# test_all_labels.pop(0)
# test_all_probs_ones.pop(0)
# test_all_patients = np.array(test_all_patients, dtype=np.float32)
# test_all_labels = np.array(test_all_labels, dtype=np.float32)
# test_all_probs_ones = np.array(test_all_probs_ones, dtype=np.float32)


# testpatients = []
# for p in range(len(test_all_patients)):
#     if not (test_all_patients[p] in testpatients):
#         testpatients.append(test_all_patients[p])
# print("distinct patients:", testpatients)

# testpatient_ave_preds = []
# testpatientlabels = []
# testcount = 0
# testsum_pat_pred = 0
# testcur_pat_labels = []
# testpatientwise_auroc = 0


# while(testcount < len(testpatients)):
#     for p in range(len(test_all_patients)):
#         if (test_all_patients[p] == testpatients[testcount]): #one patient at a time
#             testsum_pat_pred += test_all_probs_ones[p]
#             testcur_pat_label = test_all_labels[p]
#             testcur_pat_labels.append(test_all_labels[p])

#     testpatient_ave_preds.append(testsum_pat_pred / float(len(testcur_pat_labels)))
#     testpatientlabels.append(testcur_pat_labels[0])
#     #print("patient average prediction:", patient_ave_preds[count], "patient's labels:", cur_pat_labels, "length patient:", len(cur_pat_labels))

#     testcount += 1
#     testcur_pat_labels = []
#     testsum_pat_pred = 0

# print('thresh %.5f\nthresh tpr weighted %.5f\nthresh fpr weighted %.5f' % (args.best_thresh, args.best_thresh_2tpr, args.best_thresh_2fpr))
# thresh = args.best_thresh
# print("THRESHOLD:", thresh)

# acc = []
# test_corrects = 0.0
# threshlabels = []
# threshlabel = 0

# for pp in range(len(testpatient_ave_preds)):
#     if (testpatient_ave_preds[pp] >= thresh):
#         threshlabel = 1
#     else:
#         threshlabel = 0
#     threshlabels.append(threshlabel)
#     if(threshlabel == testpatientlabels[pp]):
#         test_corrects += 1.0
# acc = test_corrects / len(testpatient_ave_preds)
# test_corrects = 0.0

# acc2 = []
# test_corrects2 = 0.0
# threshlabelstpr = []
# threshlabeltpr = 0
# for pp in range(len(testpatient_ave_preds)):
#     if (testpatient_ave_preds[pp] >= args.best_thresh_2tpr):
#         threshlabeltpr = 1
#     else:
#         threshlabeltpr = 0
#     threshlabelstpr.append(threshlabeltpr)
#     if(threshlabeltpr == testpatientlabels[pp]):
#         test_corrects2 += 1.0
# acc2 = test_corrects2 / len(testpatient_ave_preds)
# test_corrects2 = 0.0


# acc3 = []
# test_corrects3 = 0.0
# threshlabelsfpr = []
# threshlabelfpr = 0
# for pp in range(len(testpatient_ave_preds)):
#     if (testpatient_ave_preds[pp] >= args.best_thresh_2fpr):
#         threshlabelfpr = 1
#     else:
#         threshlabelfpr = 0
#     threshlabelsfpr.append(threshlabelfpr)
#     if(threshlabelfpr == testpatientlabels[pp]):
#         test_corrects3 += 1.0
# acc3= test_corrects3 / len(testpatient_ave_preds)
# test_corrects3 = 0.0


# #CONFUSION MATRIX
# truepos = 0
# falsepos = 0
# trueneg = 0
# falseneg = 0
# truepospats = []
# falsepospats = []
# truenegpats5 = []
# falsenegpats = []

# for l in range(len(testpatientlabels)):
#     lab = testpatientlabels[l]
#     plab = threshlabels[l]
#     if (lab == 1):
#         if (plab == 1):
#             truepos += 1
#             truepospats.append(testpatients[l])
#         if (plab == 0):
#             falseneg += 1
#             falsenegpats.append(testpatients[l])
#     else:
#         if (plab == 0):
#             trueneg += 1
#             if(len(truenegpats5)<5 or testpatients[l] == "188_"):
#                 truenegpats5.append(testpatients[l])
#         if (plab == 1):
#             falsepos += 1
#             falsepospats.append(testpatients[l])

            
# print("confusion matrix")
#  #                   (actually pos)    (actually neg)
# #  (predicted pos)     True Pos          False Pos
# #  (predicted neg)     False Neg         True Neg

# print("true pos:", truepos, "     false pos:", falsepos, "\nfalse neg:", falseneg, "    true neg:", trueneg)

# if (truepos > 0):
#     precision = (truepos) / (truepos + falsepos) #how many of the predicted positives were actually malignant
#     recall = truepos / (truepos + falseneg) #how many of the malignants were predicted as malignant
#     f1 = (2 * precision * recall) / (precision + recall) #harmonic mean
# else:
#     precision = 0
#     recall = 0
#     f1 = 0
# accuracy = (truepos + trueneg) / (len(testpatientlabels))
# print('precision: %.4f, recall: %.4f\naccuracy: %.4f, f1: %.4f\n' % (precision, recall, accuracy, f1))

# #calculate auroc based on average score for each patient
# testpatientwise_auroc = roc_auc_score(testpatientlabels, testpatient_ave_preds)
# print("\n\nTEST Total AUROC by patient-wise average predictions:", testpatientwise_auroc)

# print("\nTest: THRESHOLD PREDICTIONS VS. LABELS!")
# for pp in range(len(testpatientlabels)):
#     print("PATIENT:", testpatients[pp], "prediction:", threshlabels[pp], "label:", testpatientlabels[pp], "prob:", testpatient_ave_preds[pp])

# plt.scatter(testpatientlabels, testpatient_ave_preds)
# plt.xlabel("Label")
# plt.ylabel("probability")

# print('done')
# print("true pos patients:", truepospats, "\nfalse neg patients:", falsenegpats)

# #save test output predictions
# testsavefile = "saved_cnn_testcv" + str(args.cvphase) + "_adj.csv"

# f=open(testsavefile,'w', newline ='\n')
# count = 0
# f.write("annot_id, label, probability, prediction from thresh "+str(args.best_thresh)+", prediction from thresh 2tpr "+str(args.best_thresh_2tpr)+", prediction from thresh 2fpr "+str(args.best_thresh_2fpr)+"\n") #titles

# for i,j,k,l,m,n in zip(testpatients, testpatientlabels, testpatient_ave_preds, threshlabels, threshlabelstpr, threshlabelsfpr):
#     if (count % 200 == 0):
#         print(i, j, k, l, m, n)
#     f.write(str(i)) #annot_id
#     f.write("," + str(int(j))) #label
#     f.write("," + str(k))
#     f.write("," + str(int(l)))
#     f.write("," + str(int(m)))
#     f.write("," + str(int(n)))
#     if(count%10 == 0):
#         print(str(i))
#     f.write("\n")
#     count += 1
# f.close()
