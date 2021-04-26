from cnn_data_augmentations import * #augmentations are not functions so not sure what this is doing
#import mobilenet_preprocess
from mobilenet_dataset import *
import model_setup #has setup_model and save_networks functions

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
#from sklearn.datasets import make_classification

import csv

def roc_auc_score_FIXED(y_true, y_pred):
    if (len(np.unique(y_true)) == 1): # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)

def focal_loss(bce_loss, targets, gamma, alpha):
    """Binary focal loss, mean. Per https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5 with improvements for alpha.
     :param bce_loss: Binary Cross Entropy loss, a torch tensor.
     :param targets: a torch tensor containing the ground truth, 0s and 1s.
     :param gamma: focal loss power parameter, a float scalar.
     :param alpha: weight of the class indicated by 1, a float scalar."""
    p_t = torch.exp(-bce_loss)
    
    #L=−αt(1−pt)γlog(pt)
    alpha_tensor = (1 - alpha) + targets * (2 * alpha - 1) #alpha if target = 1 and 1 - alpha if target = 0
    
    f_loss = alpha_tensor * (1 - p_t) ** gamma * bce_loss
    return f_loss.mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(args):
    n_epochs_stop = 5
    weightdecval = 0.003
    
    seed_value = 1
    torch.manual_seed(seed_value)

    #learning rate
    args.lr = 0.001
    floss_alpha = 0.9
    floss_gamma = 2.4

    losses = []
    f_losses = []
    losses_val = []
    f_losses_val = []

    total_all = 0
    correct_all = 0

    all_labels = []
    all_probs_ones = []
    all_patients = []
    all_preds = []

    #learning rate
    lrs = []
    
    correct = 0
    total = 0
    valcount = 0.0
    traincount = 0.0
    traincorrect = 0.0
    traintotal = 0.0
    
    running_loss = 0.0
    running_focal_loss = 0.0
    running_loss_val = 0.0
    running_focal_loss_val = 0.0
    
    tlabelsnp = np.zeros(16)
    epoch_aurocs = []

    floss_alpha = 0.9
    floss_gamma = 2.4

    #default for early stopping
    min_val_loss = 10
    prev_val_loss = 10
    epochs_no_improve = 0
    early_stop = False
    min_epoch = 0

    #max epochs
    if(args.trainval):
        args.num_epochs = args.best_epoch+2
    else:
        args.num_epochs = 100

    args.weighted_sampler = True
    
    
    #############DONE WITH DEFINING VARIABLES
    ###############NOW DEFINE MODEL
    
    training_directory = "{0}".format(args.project_home_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    os.chdir(training_directory)

    if(args.modeltype == 'mobilenet'):
        model = models.mobilenet_v2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = True
        model.classifier._modules['1'] = nn.Linear(1280, 256)
        model.classifier._modules['2'] = nn.Linear(256, 2)

    print(args.modeltype, "model:")#, model.classifier)

    #FREEZE SOME LAYERS
    ct = 0
    xfrozen = 0
    for child in model.children():
        #print("CHILD:",child)
        for kid in child.children():
            #freeze first xfrozen layers:
            if (ct < xfrozen):
                for param in kid.parameters():
                    param.requires_grad = False
                print("KID frozen", kid)
            else:
                print("not frozen:\n", kid)
            ct += 1
    print('trainable parameters', count_parameters(model))
    
    print("batch size: {}, learning rate: {}, output classes: {}, optimizer: SGD".format(args.batchSize, args.lr, args.num_classes))

    #moves to gpu
    model = model.to(device)
    
    if(not args.trainval):
        args.model_dir = args.modeltype + '_thyroid_weighted_focal_extralinear_adj_trainonly'
        args.model_path = args.project_home_dir + 'model/' + args.model_dir + '/'
    else: #trainval
        args.model_dir = args.modeltype + '_thyroid_weighted_focal_extralinear_adj'
        args.model_path = args.project_home_dir + 'model/' + args.model_dir + '/'
    
    args.phase = "train"
    model_setup.setup_model(model, "train", args.cvphase, args.best_epoch, args.pretrained_dir, args)
    print(args.modeltype, "Model Construction Complete")


    #adding weights to loss function because of imbalance in dataset
    criterion = nn.CrossEntropyLoss().to(device)
        
    if(args.weightdecay):
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay = weightdecval, momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if(args.weighted_sampler):
        rowcount = 0
        with open(args.project_home_dir + "samplesweight.csv", newline='') as infh:
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
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(samples_weight), eta_min=0.00001)    
    
    
    #load DATA
    if(not args.trainval):
        train_set = DatasetThyroid3StackedNew(args, "train", args.cvphase, args.frametype, transform=transformAug)
    else:
        trainval_set = DatasetThyroid3StackedNew(args, "trainval", args.cvphase, args.frametype, transform=transformAug)
    val_set = DatasetThyroid3StackedNew(args, "val", args.cvphase, args.frametype, transform=transformNorm)

    
    ####START TRAINING!!!!
    for epoch in range(args.num_epochs):

        #DOING THIS BY EPOCH FOR AUROC
        all_labels = []
        all_probs_ones = []
        all_patients = []

        epochstart = time.time()
        print("\n\nEPOCH", epoch, ":\n")
        correct = 0
        total = 0
        traincorrect = 0
        traintotal = 0
        valcount = 0.0
        traincount = 0.0
        
        if(args.weighted_sampler):
            #weighted sampler:
            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
            if(args.trainval):
                train_set_loader = DataLoader(dataset=trainval_set, num_workers=0, batch_size=args.batchSize, sampler=sampler, shuffle=False)
            else:
                train_set_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=args.batchSize, sampler=sampler, shuffle=False)

        else:
            #no weighted sampler!
            if(args.trainval):
                train_set_loader = DataLoader(dataset=trainval_set, num_workers=0, batch_size=args.batchSize, shuffle=True)
            else:
                train_set_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=args.batchSize, shuffle=True)
        
        if(epoch%10 == 0):
            print('Loading total', len(train_set), 'training images--------')
        
        val_set_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=args.batchSize, shuffle=False)
        if(epoch%10 == 0):
            print('Loading total', len(val_set), 'val images--------')

        
        #UPDATE LEARNING RATE ONCE PER EPOCH
        if(epoch>0 and (epoch%10 == 0)):
            scheduler.step()
            print("Learning rate at epoch", epoch, "is:", optimizer.param_groups[0]['lr'])#, "aka", scheduler.get_last_lr())
        else:
            print("Learning rate at epoch", epoch, "is:", args.lr, "aka", optimizer.param_groups[0]['lr'])
        lrs.append(optimizer.param_groups[0]['lr'])
        
        #this calls getitem (for each i in train_set_loader)
        print("num iterations of train:", len(train_set_loader))
        model.train() #train mode
        for i, data in enumerate(train_set_loader):
            inputs = data['input'].to(device)
            labels = data['label'].to(device)
            annot_ids = data['annot_id']
            frame_nums = data['frame_num']
            
            optimizer.zero_grad() #no accumulated gradients from other batches

            if (i == 0):
                print("batch index {}, 0/1 distribution: {}/{}".format(i, len(np.where(labels.cpu().numpy() == 0)[0]),
            len(np.where(labels.cpu().numpy() == 1)[0])))

            # forward + backward + optimize (to find good parameters: weights + bias)
            outputs = model(inputs).to(device)
            labels = labels.squeeze(1)

            try:
                loss = criterion(outputs, labels)
                f_loss = focal_loss(loss, labels, floss_gamma, floss_alpha)
            except:
                print("1train loss failed", inputs.shape, outputs.shape, labels.shape)
            try:
                #loss.backward()
                f_loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_focal_loss += f_loss.item()
                traincount += 1.0
            except:
                print("2train loss failed", inputs.shape, outputs.shape, labels.shape)

            #TRAIN ACCURACY
            if (args.probFunc == 'Softmax'):
                sf = nn.Softmax(dim=1) #makes items in a row add to 1; dim = 0 makes items in a column add to 1
                outputs = sf(outputs)

            _, trainpredicted = torch.max(outputs, 1)
            try:
                if torch.cuda.is_available():
                    traincorrect += (trainpredicted.cpu() == labels.cpu()).sum()#(thresh_predicted == tlabels.cpu()).sum()
                else:
                    traincorrect += (trainpredicted == labels).sum()
            except:
                print("??")
                traincorrect += (trainpredicted == labelsnp).sum()

            # Total number of labels
            try:
                traintotal += len(labels.cpu())#tlabels.size(0)
            except:
                print(i, "index len 0 trainlabels; input shape", np.shape(inputs.cpu().numpy()))

            trainaccuracy = 100 * traincorrect // traintotal

            
            x = 100
            if (i % x == 0):
                #validation phase of model
                model.eval()

                #calculate val accuracy
                with torch.no_grad():

                    for j, dataa in enumerate(val_set_loader):
                        tinputs = dataa['input'].to(device)
                        tlabels = dataa['label'].to(device)
                        #new
                        tannot_ids = dataa['annot_id']
                        tframe_nums = dataa['frame_num']


                        # Forward pass only to get logits/output
                        outs = model(tinputs)
                        tlabels = tlabels.squeeze(1)#tlabels.squeeze()

                        # Get predictions from the maximum value                        
                        if (args.probFunc == 'Softmax'):
                            sf = nn.Softmax(dim=1) #makes items in a row add to 1; dim = 0 makes items in a column add to 1
                            outs = sf(outs)
                        elif (args.probFunc == 'Sigmoid'):
                            sf = nn.Sigmoid()
                            outs = sf(outs)

                        _, predicted = torch.max(outs, 1)

                        #add labels + predictions to full set for auroc later
                        if torch.cuda.is_available():
                            outs_ones = outs.detach().cpu().numpy()[:, 1]
                            tlabelsnp = tlabels.cpu().numpy()
                            all_labels = np.append(all_labels, tlabelsnp)
                        else:
                            outs_ones = outs.detach().numpy()[:, 1]
                            tlabels = tlabels.numpy()
                            all_labels = np.append(all_labels, tlabels)

                        all_probs_ones = np.append(all_probs_ones, outs_ones)

                        tannot_ids = np.asarray(tannot_ids)
                        all_patients = np.append(all_patients, tannot_ids)


                        if(j%50 == 0):
                            try:
                                #val loss
                                loss_val = criterion(outs, tlabels)
                                f_loss_val = focal_loss(loss_val, tlabels, floss_gamma, floss_alpha)

                                running_loss_val += loss_val.item()
                                running_focal_loss_val += f_loss_val.item()
                                valcount += 1
                            except:
                                print("val loss failed\n\n")


                        #  USE GPU FOR MODEL
                        # Total correct predictions
                        try:
                            if torch.cuda.is_available():
                                correct += (predicted.cpu() == tlabels.cpu()).sum()#(thresh_predicted == tlabels.cpu()).sum()
                            else:
                                correct += (predicted == tlabels).sum()
                        except:
                            print("??")
                            correct += (predicted == tlabelsnp).sum()

                        # Total number of labels
                        try:
                            total += len(tlabels.cpu())#tlabels.size(0)
                        except:
                            print(j, "index len 0 tlabels; input shape", np.shape(tinputs.cpu().numpy()))

                        #print("total", total)
                        accuracy = 100 * correct // total

                        #end of torch no grad


                #print statistics
                message = 'epoch: %d, iters: %d, time: %.3f, with loss: %.3f, focal weighted loss: %.5f, val loss: %.3f, val focal weighted loss: %.5f, train_acc: %.4f, val accuracy: %.3f' % (
                    epoch, i + 1, time.time() - epochstart, loss, f_loss, loss_val, f_loss_val, trainaccuracy.data, accuracy.data)
                print(message)
                
                model.train() #back to train mode (done with validation)
                #end of validation mode

        #end of epoch
        total_all += total
        correct_all += correct

        losses.append(running_loss / traincount)#len(train_set))
        f_losses.append(running_focal_loss / traincount)#len(train_set))

        val_loss = running_loss_val / valcount

        losses_val.append(val_loss)
        f_losses_val.append(running_focal_loss_val / valcount)
        print("Epoch", epoch, "overall val loss:", val_loss, "overall train loss:", (running_loss / traincount), "or train over len trainset?", (running_loss / len(train_set)))

        if(not args.trainval):
            if ((val_loss < prev_val_loss) and epoch != 0): #current epoch val loss improved
          # Save the model #torch.save(model)
                epochs_no_improve = 0
                if (val_loss < min_val_loss and epoch>30):
                    min_val_loss = val_loss
                    min_epoch = epoch
                prev_val_loss = val_loss
                early_stop = False
            else:
                print("val loss did not improve vs. last epoch", prev_val_loss, ", epochs not improving =", epochs_no_improve, "... min is", min_val_loss, "from epoch", min_epoch)
                epochs_no_improve += 1
                prev_val_loss = val_loss

            if (epoch > 40 and epochs_no_improve >= n_epochs_stop):
                print('Early stopping! end of epoch', epoch, ", val loss =", val_loss, "min val loss =", min_val_loss, "from epoch", min_epoch)
                early_stop = True


        running_loss = 0.0
        running_focal_loss = 0.0
        running_loss_val = 0.0
        running_focal_loss_val = 0.0
        valcount = 0.0
        traincount = 0.0

        patients = []
        for p in range(len(all_patients)):
            if not (all_patients[p] in patients):
                patients.append(all_patients[p])

        patient_ave_preds = []
        patientlabels = []
        count = 0
        sum_pat_pred = 0
        cur_pat_labels = []
        patientwise_auroc = 0

        while(count < len(patients)):
            for p in range(len(all_patients)):
                if (all_patients[p] == patients[count]): #one patient at a time
                    sum_pat_pred += all_probs_ones[p]
                    cur_pat_label = all_labels[p]
                    cur_pat_labels.append(all_labels[p])

            patient_ave_preds.append(sum_pat_pred / float(len(cur_pat_labels)))
            patientlabels.append(cur_pat_labels[0])

            count += 1
            cur_pat_labels = []
            sum_pat_pred = 0

        #calculate auroc for epoch based on each image score
        try:
            full_auroc = roc_auc_score(all_labels, all_probs_ones)
            print("AUROC", full_auroc)
        except:
            full_auroc = roc_auc_score_FIXED(all_labels, all_probs_ones)
            print("AUROC", full_auroc)
            
        #calculate auroc for epoch based on average score for each patient
        patientwise_auroc = roc_auc_score(patientlabels, patient_ave_preds)
        print("Epoch", epoch,"AUROC by patient-wise average predictions =", patientwise_auroc, "\n\n")
        epoch_aurocs.append(patientwise_auroc)


        if (early_stop):
            print("Training Stopped")
            break

        save_freq = 10
        if ((epoch % save_freq == 0) or (args.trainval and epoch >= (args.best_epoch-2))):
            print('saving the latest model (epoch %d) of learning rate %f' % (epoch, args.lr))
            
            if(not args.trainval):
                args.model_dir = args.modeltype + '_thyroid_weighted_focal_extralinear_adj_trainonly'
                args.model_path = args.project_home_dir + 'model/' + args.model_dir + '/'
            else: #trainval
                args.model_dir = args.modeltype + '_thyroid_weighted_focal_extralinear_adj'
                args.model_path = args.project_home_dir + 'model/' + args.model_dir + '/'
            print("to:", args.model_path)
            model_setup.save_networks(model, epoch, args.cvphase, args.model_path, args.model_dir)
    
    try:
        full_auroc = roc_auc_score(all_labels, all_probs_ones)
        print("AUROC", full_auroc)
    except:
        full_auroc = roc_auc_score_FIXED(all_labels, all_probs_ones)
        print("AUROC", full_auroc)


    patients = []
    for p in range(len(all_patients)):
        if not (all_patients[p] in patients):
            patients.append(all_patients[p])
    print("distinct patients:", patients)

    patient_ave_preds = []
    patientlabels = []
    count = 0
    sum_pat_pred = 0
    cur_pat_labels = []

    while(count < len(patients)):
        for p in range(len(all_patients)):
            if (all_patients[p] == patients[count]): #one patient at a time
                sum_pat_pred += all_probs_ones[p]
                cur_pat_label = all_labels[p]
                cur_pat_labels.append(all_labels[p])

        patient_ave_preds.append(sum_pat_pred / float(len(cur_pat_labels)))
        patientlabels.append(cur_pat_labels[0])

        count += 1
        cur_pat_labels = []
        sum_pat_pred = 0
        
        
#     if(not args.trainval): #ONLY train phase get threshold
#         fpr, tpr, thresholds = roc_curve(patientlabels, patient_ave_preds)
#         # get the best threshold
#         J = tpr - fpr
#         J2 = (2*tpr) - fpr
#         J3 = tpr - (2*fpr)
#         ix = argmax(J) #index of max value
#         ix2 = argmax(J2)
#         ix3 = argmax(J3)
#         args.best_thresh = thresholds[ix]
#         args.best_thresh_2tpr = thresholds[ix2]
#         args.best_thresh_2fpr = thresholds[ix3]
#         print('Best Threshold=%f, weighted tpr Threshold=%f, weighted fpr Threshold=%f' % (args.best_thresh, args.best_thresh_2tpr, args.best_thresh_2fpr),)
#         print('index 1=%f, weighted tpr index=%f, weighted fpr index=%f' % (ix, ix2, ix3),)

    print("\nall average patient predictions and their labels:")
    for hh in range(len(patientlabels)):
        print("label:", patientlabels[hh], "prediction:", patient_ave_preds[hh])
    #calculate auroc based on average score for each patient
    patientwise_auroc = roc_auc_score(patientlabels, patient_ave_preds)
    print("\n\nTotal AUROC by patient-wise average predictions:", patientwise_auroc)
    
    if(not args.trainval):
        print("Min validation loss epoch:", min_epoch, ", loss =", min_val_loss, ", train loss =", losses[min_epoch], ", val auroc =", epoch_aurocs[min_epoch])
    else:
        print("Min train 3/5 validation loss epoch:", args.best_epoch, ", loss =", min_val_loss)
    
    plt.plot(lrs)
    print(lrs)
    plt.xlabel("Epochs")
    plt.ylabel("Learning rates")
    plt.show()
    
    plot_test_stats(losses, f_losses, losses_val, f_losses_val, epoch_aurocs, patientlabels, patient_ave_preds)
    calc_test_stats(patients, patientlabels, patient_ave_preds)
    
    return min_epoch

