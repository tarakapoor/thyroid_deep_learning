import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import csv

from numpy import sqrt
from numpy import argmax

from main import configread, configwrite


def plot_test_stats(losses, f_losses, losses_val, f_losses_val, epoch_aurocs, patientlabels, patient_ave_preds):
    config = configread()

    epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    if(config['trainval']):
        print("train losses (4/5 data crossvalidation)")
    else:
        print("train losses 3/5 data")
    plt.plot(losses)
    print(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train CrossEntropy Loss")
    plt.show()

    print("val losses")
    plt.plot(losses_val)
    print(losses_val)
    plt.ylabel("Val CrossEntropy Loss")
    plt.show()

    print("train vs. val losses")
    y1 = losses_val
    plt.plot(y1, label = "val losses")
    y2 = losses
    plt.plot(y2, label = "train losses")
    plt.legend()
    plt.show()

    print("train vs. val focal losses")
    plt.plot(f_losses_val, label = "val focal losses")
    plt.plot(f_losses, label = "train focal losses")
    plt.legend()
    plt.show()

    print("aurocs by epoch")
    plt.plot(epoch_aurocs)
    print(epoch_aurocs)
    plt.ylabel("Validation AUROC")
    plt.show()
    
    #########
    if (not config['trainval']): #thresholds from first train phase
        x = [0, 0.01, 0.02, 0.03, 0.09, 0.16, 0.116, 0.126, 0.2, .35, 0.351, 0.352, 0.353, 0.39, 0.3916, 0.392, .4, .41, .42, .43, 0.5, 0.6, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.671, 0.672, 0.673, 0.68, 0.681, 0.682, 0.8, 1.0]
        fpr, tpr, thresholds = roc_curve(patientlabels, patient_ave_preds)

        print("VALIDATION ROC")
        plt.plot(fpr, tpr)
        plt.plot(x, x)
        plt.xlabel("False Positives")
        plt.ylabel("True Positives")
        plt.show()

        sens = tpr
        spec = 1 - fpr
        gmean = []
        for x in range(len(sens)):
            gmean.append(math.sqrt(sens[x]*spec[x]))
        bestgmean = argmax(gmean)
        gmeanthresh = thresholds[bestgmean]

        best = argmax(tpr - fpr)
        thresh = thresholds[best]
        config = configwrite('best_thresh', thresh)
        
        best2tpr = argmax(2*tpr - fpr)
        thresh2tpr = thresholds[best2tpr]
        config = configwrite('best_thresh_2tpr', thresh2tpr)
        
        best2fpr = argmax(tpr - 2*fpr)
        thresh2fpr = thresholds[best2fpr]
        config = configwrite('best_thresh_2fpr', thresh2fpr)
        
        best = argmax(tpr - fpr)
        thresh = thresholds[best]

        plt.xlabel("Threshold")
        tprfpr, = plt.plot(thresholds, tpr - fpr, label="True Positives - False Positives")
        gmeannn, = plt.plot(thresholds, gmean, label="GMean")
        plt.legend([tprfpr, gmeannn], ["True Positives - False Positives", "GMean"])
        print("VALIDATION thresholds")
        plt.show()
        
    print("thresholds: unweighted and fpr weighted and tpr weighted", config['best_thresh'], config['best_thresh_2fpr'], config['best_thresh_2tpr'])
    
    
    

def calc_test_stats(distinct_patients, patientlabels, patient_ave_preds):
    config = configread()

    #calculate auroc based on average score for each patient
    patientwise_auroc = roc_auc_score(patientlabels, patient_ave_preds)
    print("\nTotal AUROC by patient-wise average predictions:", patientwise_auroc)

    #accuracy based on threshold predictions
    threshlabels = []
    for pp in range(len(patient_ave_preds)):
        if (patient_ave_preds[pp] >= config['best_thresh']):
            print(distinct_patients[pp], patient_ave_preds[pp])
            threshlabels.append("1")
        else:
            threshlabels.append("0")

    print("\n\nTHRESHOLD PREDICTIONS VS. LABELS!")
    print("threshold probability:", config['best_thresh'])
    for pp in range(len(patientlabels)):
        print("PATIENT:", distinct_patients[pp], "prediction:", threshlabels[pp], "label:", patientlabels[pp], "prob:", patient_ave_preds[pp])

    plt.scatter(patientlabels, patient_ave_preds)
    plt.xlabel("Label")
    plt.ylabel("probability")
    print('done, best epoch:', config['best_epoch'])

    

def analyze_test_outputs(savefilename, modeltype): #modeltype cnn or transformer
    config = configread()

    #ANALYSIS OF RESULTS
    print("CV PHASE", config['cvphase'])

    test_all_labels = []
    test_all_patients = []
    test_all_probs_ones = []
    rowcount = 0
    testsaveallfile = savefilename#"test_all_outs" + str(config['cvphase) + ".csv"
    with open(testsaveallfile, newline='') as infh:
        print("opened csv for test outputs")
        reader = csv.reader(infh)
        rowcount = 0
        for row in reader:
            test_all_patients.append(row[0])
            test_all_labels.append(row[1])
            test_all_probs_ones.append(row[2])
            if(rowcount % 500 == 0):
                print("row:", rowcount)
            rowcount += 1
        print("rowcount", rowcount)
    test_all_patients.pop(0)
    test_all_labels.pop(0)
    test_all_probs_ones.pop(0)
    test_all_labels = np.array(test_all_labels, dtype=np.float32)
    test_all_probs_ones = np.array(test_all_probs_ones, dtype=np.float32)

    testpatients = []
    for p in range(len(test_all_patients)):
        if not (test_all_patients[p] in testpatients):
            testpatients.append(test_all_patients[p])
    print("distinct patients:", testpatients)

    testpatient_ave_preds = []
    testpatientlabels = []
    testcount = 0
    testsum_pat_pred = 0
    testcur_pat_labels = []
    testpatientwise_auroc = 0

    while(testcount < len(testpatients)):
        for p in range(len(test_all_patients)):
            if (test_all_patients[p] == testpatients[testcount]): #one patient at a time
                testsum_pat_pred += test_all_probs_ones[p]
                testcur_pat_label = test_all_labels[p]
                testcur_pat_labels.append(test_all_labels[p])

        testpatient_ave_preds.append(testsum_pat_pred / float(len(testcur_pat_labels)))
        testpatientlabels.append(testcur_pat_labels[0])
        testcount += 1
        testcur_pat_labels = []
        testsum_pat_pred = 0

    print('thresh %.8f\nthresh fpr weighted %.5f\nthresh tpr weighted %.8f' % (config['best_thresh'], config['best_thresh_2fpr'], config['best_thresh_2tpr']))
    thresh = config['best_thresh_2fpr'] #using false positive threshold now for confusion matrix
    print('CHOSEN THRESHOLD: %.7f' % (thresh))
    
    acc = []
    test_corrects = 0.0
    threshlabels = []
    threshlabel = 0

    for pp in range(len(testpatient_ave_preds)):
        if (testpatient_ave_preds[pp] >= thresh):
            threshlabel = 1
        else:
            threshlabel = 0
        threshlabels.append(threshlabel)
        if(threshlabel == testpatientlabels[pp]):
            test_corrects += 1.0
    acc = test_corrects / len(testpatient_ave_preds)
    test_corrects = 0.0

    acc2 = []
    test_corrects2 = 0.0
    threshlabelstpr = []
    threshlabeltpr = 0
    for pp in range(len(testpatient_ave_preds)):
        if (testpatient_ave_preds[pp] >= config['best_thresh_2tpr']):
            threshlabeltpr = 1
        else:
            threshlabeltpr = 0
        threshlabelstpr.append(threshlabeltpr)
        if(threshlabeltpr == testpatientlabels[pp]):
            test_corrects2 += 1.0
    acc2 = test_corrects2 / len(testpatient_ave_preds)
    test_corrects2 = 0.0

    acc3 = []
    test_corrects3 = 0.0
    threshlabelsfpr = []
    threshlabelfpr = 0
    for pp in range(len(testpatient_ave_preds)):
        if (testpatient_ave_preds[pp] >= config['best_thresh_2fpr']):
            threshlabelfpr = 1
        else:
            threshlabelfpr = 0
        threshlabelsfpr.append(threshlabelfpr)
        if(threshlabelfpr == testpatientlabels[pp]):
            test_corrects3 += 1.0
    acc3 = test_corrects3 / len(testpatient_ave_preds)
    test_corrects3 = 0.0


    #CONFUSION MATRIX
    truepos = 0
    falsepos = 0
    trueneg = 0
    falseneg = 0
    truepospats = []
    falsepospats = []
    truenegpats5 = []
    falsenegpats = []

    for l in range(len(testpatientlabels)):
        lab = testpatientlabels[l]
        plab = threshlabels[l]
        if (lab == 1):
            if (plab == 1):
                truepos += 1
                truepospats.append(testpatients[l])
            if (plab == 0):
                falseneg += 1
                falsenegpats.append(testpatients[l])
        else:
            if (plab == 0):
                trueneg += 1
                if(len(truenegpats5)<5):
                    truenegpats5.append(testpatients[l])
            if (plab == 1):
                falsepos += 1
                falsepospats.append(testpatients[l])

    print("confusion matrix")
     #                   (actually pos)    (actually neg)
    #  (predicted pos)     True Pos          False Pos
    #  (predicted neg)     False Neg         True Neg
    print("true pos:", truepos, "     false pos:", falsepos, "\nfalse neg:", falseneg, "    true neg:", trueneg)

    if (truepos > 0):
        precision = (truepos) / (truepos + falsepos) #how many of the predicted positives were actually malignant
        recall = truepos / (truepos + falseneg) #how many of the malignants were predicted as malignant
        f1 = (2 * precision * recall) / (precision + recall) #harmonic mean
    else:
        precision = 0
        recall = 0
        f1 = 0
    accuracy = (truepos + trueneg) / (len(testpatientlabels))
    print('precision: %.4f, recall: %.4f\naccuracy: %.4f, f1: %.4f\n' % (precision, recall, accuracy, f1))

    #calculate auroc based on average score for each patient
    testpatientwise_auroc = roc_auc_score(testpatientlabels, testpatient_ave_preds)
    print("\n\nTEST Total AUROC by patient-wise average predictions:", testpatientwise_auroc)

    print("\nTest: THRESHOLD PREDICTIONS VS. LABELS!")
    for pp in range(len(testpatientlabels)):
        print("PATIENT:", testpatients[pp], "prediction:", threshlabels[pp], "label:", testpatientlabels[pp], "prob:", testpatient_ave_preds[pp])

    plt.scatter(testpatientlabels, testpatient_ave_preds)
    plt.xlabel("Label")
    plt.ylabel("probability")
    print('Done with analysis')
    print("True positive patients:", truepospats, "\nFalse negative patients:", falsenegpats)
    
    #CONFIDENCE INTERVAL
    print("Confidence interval:")
    bootstrap_auc(testpatientlabels, testpatient_ave_preds)

    
    #WRITE OUTPUT PROBABILITIES, LABELS, and THRESHOLD PREDICTIONS BY PATIENT TO CSV FILE
    testsavefile = "saved_test_" + str(modeltype) + str(config['cvphase']) + "_adj.csv"
    print("writing outputs to csv", testsavefile)
    print("threshold", thresh)
    f=open(testsavefile,'w', newline ='\n')
    count = 0
    f.write("annot_id, label, probability, prediction from thresh "+str(config['best_thresh'])+", prediction from thresh 2tpr "+str(config['best_thresh_2tpr'])+", prediction from thresh 2fpr "+str(config['best_thresh_2fpr'])+"\n") #titles

    for i,l,j,k,m,n in zip(testpatients, testpatientlabels, testpatient_ave_preds, threshlabels, threshlabelstpr, threshlabelsfpr):
        if (count % 200 == 0):
            print(i, l, j, k, m, n)
        f.write(str(i)) #annot_id
        f.write("," + str(int(l))) #label
        f.write("," + str(j))
        f.write("," + str(int(k)))
        f.write("," + str(int(m)))
        f.write("," + str(int(n)))
        if(count%10 == 0):
            print(str(i))
        f.write("\n")
        count += 1
    f.close()
    
    

def bootstrap_auc(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    n_bootstraps = n_bootstraps
    rng_seed = rng_seed
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(len(y_pred), size=len(y_pred))
        y_true_ind = [y_true[i] for i in indices]
        y_pred_ind = [y_pred[ind] for ind in indices]
        
        if len(np.unique(y_true_ind)) < 2:
            # We need at least one positive and one negative sample for AUROC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(y_true_ind, y_pred_ind)
        bootstrapped_scores.append(score)
    bootstrapped_scores = np.array(bootstrapped_scores)

    print("AUROC: {:0.6f}".format(roc_auc_score(y_true, y_pred)))
    print("Confidence interval for the AUROC score: [{:0.5f} - {:0.5}]".format(
        np.percentile(bootstrapped_scores, (2.5, 97.5))[0], np.percentile(bootstrapped_scores, (2.5, 97.5))[1]))
    
    return roc_auc_score(y_true, y_pred), np.percentile(bootstrapped_scores, (2.5, 97.5))
