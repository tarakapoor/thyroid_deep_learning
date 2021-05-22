from mobilenet_dataset import *
import model_setup #has setup_model and save_networks functions
from cnn_data_augmentations import *

from main import configwrite, configread #from cnn
from transformer_main import transfconfigwrite #for transformer

import numpy as np
import os
import time
import csv

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


def mobilenet_model(imgpath, maskpath, labelpath, project_home_dir, pretrained_dir, min_epoch):
    """Run the CNN model to extract features to feed into Transformer.
    Write features to output csv files.
    Run once for train and val, once for trainval and test.
    Keyword arguments:
    imgpath -- path to image data hdf5
    maskpath -- path to mask data hdf5
    labelpath -- path to label csv file
    pretrained_dir -- where the model was saved to
    min_epoch -- lowest validation loss epoch to use for feature extraction"""

    config = configread() #from cnn config file
    
    all_probs = np.zeros((0,256)) #to save for transformer inputs
    val_probs = np.zeros((0,256))
    test_probs = np.zeros((0,256))

    all_labels = []
    all_probs_ones = []
    all_patients = []
    all_preds = []
    all_framenums = []
    
    test_preds = []
    
    total = 0
    correct = 0

    val_labels = []
    val_patients = []
    val_framenums = []

    test_labels = []
    test_patients = []
    test_framenums = []
    
    ttotal = 0
    tcorrect = 0
    
    #LOAD DATA
    if(config['trainval'] == False):
        train_set = DatasetThyroid3StackedNew(imgpath, maskpath, labelpath, project_home_dir, "train", config['cvphase'], config['frametype'], transform=transformNorm)
        val_set = DatasetThyroid3StackedNew(imgpath, maskpath, labelpath, project_home_dir, "val", config['cvphase'], config['frametype'], transform=transformNorm)
        print("IN CNN FEATURE EXTRACT: len train/val sets", len(train_set), len(val_set))

    else:
        trainval_set = DatasetThyroid3StackedNew(imgpath, maskpath, labelpath, project_home_dir, "trainval", config['cvphase'], config['frametype'], transform=transformNorm)
        test_set = DatasetThyroid3StackedNew(imgpath, maskpath, labelpath, project_home_dir, "test", config['cvphase'], config['frametype'], transform=transformNorm)
        print("IN CNN FEATURE EXTRACT: len trainval/test sets",  len(trainval_set), len(test_set))

    ###############NOW DEFINE MODEL AGAIN
    seed_value = 1
    torch.manual_seed(seed_value)

    #args.num_epochs = 65
    training_directory = "{0}".format(project_home_dir)#, 'thyroid_dicoms')#'dicom_files')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    os.chdir(training_directory)
    
    model = models.mobilenet_v2(pretrained=True)
    phase = "test" #always want pretrained
    

    print("pretrained directory:", pretrained_dir)
    for param in model.parameters():
        param.requires_grad = False

    #modify actual last linear layer in the mobilenet
    model.classifier._modules['1'] = nn.Linear(1280, 256)
    model.classifier._modules['2'] = nn.Linear(256, 2)

    print("batch size: {}, output classes: {}".format(config['batchSize'], config['num_classes']))
    model_setup.setup_model(model, phase, config['cvphase'], pretrained_dir, pretrained_dir, project_home_dir, min_epoch)
    
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier
    print("model classifier:", model.classifier)
    #moves to gpu
    model = model.to(device)
    print(config['modeltype'], "Model Construction Complete")
    #####################################

    weighted_sampler = True
    if(weighted_sampler):
        rowcount = 0
        with open(project_home_dir + "samplesweight.csv", newline='') as infh:
            print("opened csv for samplesweight")
            reader = csv.reader(infh)
            rowcount = 0
            samples_weight = []
            for row in reader:
                #print(row)
                del row[-1] #empty space
                row = np.array(row).astype(np.float)
                for x in row:
                    samples_weight.append(x)
            print(len(samples_weight))

        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        #weighted sampler:
        if(config['trainval']):
            train_set_loader = DataLoader(dataset=trainval_set, num_workers=0, batch_size=config['batchSize'], sampler=sampler, shuffle=False)
        else:
            train_set_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=config['batchSize'], sampler=sampler, shuffle=False)

    else:
        #no weighted sampler!
        if(config['trainval']):
            train_set_loader = DataLoader(dataset=trainval_set, num_workers=0, batch_size=config['batchSize'], shuffle=True)
        else:
            train_set_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=config['batchSize'], shuffle=True)
        
        
    if(config['trainval']):
        test_set_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=config['batchSize'], shuffle=False)
    else:
        val_set_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=config['batchSize'], shuffle=False)

    model.eval() #test mode but on training dataset before transformer
    with torch.no_grad():
        for i, data in enumerate(train_set_loader):
            inputs = data['input'].to(device)
            labels = data['label'].to(device)
            annot_ids = data['annot_id']
            frame_nums = data['frame_num']

            # Forward pass only to get logits/output
            outputs = model(inputs)
            labels = labels.squeeze()

            #add these raw outputs to list for transformer
            all_probs = np.concatenate((all_probs, outputs.detach().cpu().numpy()))
            if (i % 100 == 0):
                print("in train: probabilities shape for transformer (should be [xx, 2])", np.shape(all_probs))            
            
            #apply sigmoid or softmax
            if (config['probFunc'] == 'Softmax'):
                sf = nn.Softmax(dim=1) #makes items in a row add to 1; dim = 0 makes items in a column add to 1
                outputs = sf(outputs)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs, 1)
            
            #add labels + predictions to full set for auroc later
            if torch.cuda.is_available():
                outs_ones = outputs.detach().cpu().numpy()[:, 1]
                labelsnp = labels.cpu().numpy()
            else:
                outs_ones = outputs.detach().numpy()[:, 1]
                labelsnp = labels.numpy()

            all_labels = np.append(all_labels, labelsnp)
            all_probs_ones = np.append(all_probs_ones, outs_ones)

            annot_ids = np.asarray(annot_ids)
            frame_nums = np.asarray(frame_nums)

            all_patients = np.append(all_patients, annot_ids)
            all_framenums = np.append(all_framenums, frame_nums)
        

        ###################################################
        if(not config['trainval']):
            for v, vdata in enumerate(val_set_loader):
                vinputs = vdata['input'].to(device)
                vlabels = vdata['label'].to(device)
                vannot_ids = vdata['annot_id']
                vframe_nums = vdata['frame_num']

                # Forward pass only to get logits/output
                voutputs = model(vinputs)
                vlabels = vlabels.squeeze()

                #add these raw outputs to list for transformer
                val_probs = np.concatenate((val_probs, voutputs.detach().cpu().numpy()))

                #apply sigmoid or softmax
                if (config['probFunc'] == 'Softmax'):
                    sf = nn.Softmax(dim=1) #makes items in a row add to 1; dim = 0 makes items in a column add to 1
                    voutputs = sf(voutputs)

                # Get predictions from the maximum value
                _, predicted = torch.max(voutputs, 1)

                #add labels + predictions to full set for auroc later
                if torch.cuda.is_available():
                    vouts_ones = voutputs.detach().cpu().numpy()[:, 1]
                    vlabelsnp = vlabels.cpu().numpy()
                else:
                    vouts_ones = voutputs.detach().numpy()[:, 1]
                    vlabelsnp = vlabels.numpy()

                val_labels = np.append(val_labels, vlabelsnp)
                vannot_ids = np.asarray(vannot_ids)
                vframe_nums = np.asarray(vframe_nums)

                val_patients = np.append(val_patients, vannot_ids)
                val_framenums = np.append(val_framenums, vframe_nums)
                    
        
            #write to csv file: train and val
            print(len(all_patients))

            filename = project_home_dir + "outtrainaug_"+config['frametype']+"_featurescv" + str(config['cvphase']) + ".csv"
            valfile = project_home_dir + "outtvalaug_"+config['frametype']+"_featurescv" + str(config['cvphase']) + ".csv"
            transfconfig = transfconfigwrite('featurestrainfile', filename)
            transfconfig = transfconfigwrite('featuresvalfile', valfile)

            print("\n\ntrain writing features to csv")
            f=open(filename,'w', newline ='\n')
            count = 0
            for i,l,j,k in zip(all_patients, all_labels, all_probs, all_framenums):
                if (count % 200 == 0):
                    print(np.shape(i), np.shape(l), np.shape(j))
                f.write(str(k)) #framenum
                f.write("," +str(i)) #annot_id
                f.write("," + str(int(l))) #label
                abc=0
                for prprpr in j:
                    f.write("," + str(prprpr))
                    abc+=1
                if(count%500 == 0):
                    print(abc)
                f.write("\n")
                count += 1
            f.close()

            print("\n\nval writing features to csv")
            f=open(valfile,'w', newline ='\n')
            count = 0
            for i,l,j,k in zip(val_patients, val_labels, val_probs, val_framenums):
                f.write(str(k)) #frame num
                f.write("," + str(i)) #annot_id
                f.write("," + str(int(l))) #label
                abc=0
                for prprpr in j:
                    f.write("," + str(prprpr))#row.append(prprpr) #+","+str(j)
                    abc+=1
                if(count%500 == 0):
                    print(abc)
                f.write("\n")
                count += 1
            f.close()
            print("done writing train and val features to csv", filename, valfile)
        
        ###################################################
        inputs = []
        test_labels = []
        test_patients = []
        test_framenums = []
        outputs = []
        predicted = []
        labelsnp = []
        
        test_preds = []
        tpatients = []

        if(config['trainval']):
            for tt, ttdata in enumerate(test_set_loader):
                inputs = ttdata['input'].to(device)
                labels = ttdata['label'].to(device)
                annot_ids = ttdata['annot_id']
                frame_nums = ttdata['frame_num']

                # Forward pass only to get logits/output
                outputs = model(inputs)
                labels = labels.squeeze()

                #add these raw outputs to list for transformer
                test_probs = np.concatenate((test_probs, outputs.detach().cpu().numpy()))
                
                #apply sigmoid or softmax
                if (config['probFunc'] == 'Softmax'):
                    sf = nn.Softmax(dim=1) #makes items in a row add to 1; dim = 0 makes items in a column add to 1
                    outputs = sf(outputs)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs, 1)

                #add labels + predictions to full set for auroc later
                if torch.cuda.is_available():
                    outs_ones = outputs.detach().cpu().numpy()[:, 1]
                    labelsnp = labels.cpu().numpy()
                else:
                    outs_ones = outputs.detach().numpy()[:, 1]
                    labelsnp = labels.numpy()

                test_labels = np.append(test_labels, labelsnp)
                annot_ids = np.asarray(annot_ids)
                frame_nums = np.asarray(frame_nums)
                test_patients = np.append(test_patients, annot_ids)
                test_framenums = np.append(test_framenums, frame_nums)
                test_preds = np.append(test_preds, outs_ones)

                for p in range(len(test_patients)):
                    if not (test_patients[p] in tpatients):
                        tpatients.append(test_patients[p])

            tpatient_ave_preds = []
            tpatientlabels = []
            tcount = 0
            tsum_pat_pred = 0
            tcur_pat_labels = []

            #patient wise predictions aggregate
            while(tcount < len(tpatients)):
                for tp in range(len(test_patients)):
                    if (test_patients[tp] == tpatients[tcount]): #one patient at a time
                        tsum_pat_pred += test_preds[tp]
                        tcur_pat_labels.append(test_labels[tp])

                tpatient_ave_preds.append(tsum_pat_pred / float(len(tcur_pat_labels)))
                tpatientlabels.append(tcur_pat_labels[0])

                tcount += 1
                tcur_pat_labels = []
                tsum_pat_pred = 0
            
            trainvalfile = project_home_dir + "outtrainvalaug_"+config['frametype']+"_featurescv" + str(config['cvphase']) + ".csv"
            testfile = project_home_dir + "outtestaug_"+config['frametype']+"_featurescv" + str(config['cvphase']) + ".csv"
            transfconfig = transfconfigwrite('featurestrainvalfile', trainvalfile)
            transfconfig = transfconfigwrite('featurestestfile', testfile)
            print(np.shape(all_patients), np.shape(all_labels), np.shape(all_probs), np.shape(all_framenums), np.shape(test_patients), np.shape(test_labels), np.shape(test_probs))

            print("saving trainval features:")

            f=open(trainvalfile,'w', newline ='\n')
            count = 0
            for i,l,j,k in zip(all_patients, all_labels, all_probs, all_framenums):
                if (count % 200 == 0):
                    print(np.shape(i), np.shape(l), np.shape(j))
                f.write(str(k))#framenum
                f.write("," + str(i)) #annot_id
                f.write("," + str(int(l))) #label
                abc=0
                for prprpr in j:
                    f.write("," + str(prprpr))
                    abc+=1
                if(count%500 == 0):
                    print(abc)
                f.write("\n")
                count += 1
            f.close()

            print("\n\n test \ntest patients:", test_patients)
            f=open(testfile,'w', newline ='\n')
            count = 0
            for i,l,j,k in zip(test_patients, test_labels, test_probs, test_framenums):
                if (count % 200 == 0):
                    print(np.shape(i), np.shape(l), np.shape(j))
                f.write(str(k)) #frame num
                f.write("," + str(i)) #annot_id
                f.write("," + str(int(l))) #label
                abc=0
                for prprpr in j:
                    f.write("," + str(prprpr))
                    abc+=1
                if(count%500 == 0):
                    print(abc)
                f.write("\n")
                count += 1
            f.close()    
            print("done writing trainval and test features to csv", trainvalfile, testfile)
