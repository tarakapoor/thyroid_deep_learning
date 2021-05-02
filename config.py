#config file: not to be edited by user

#permanents
batchSize = 16
lr = 0.001
num_classes = 2
frametype = "adjacent"
probFunc = "Softmax"
weightdecay = False
weightdecval = 0.003 #how much weight decay (if using)
modeltype = "mobilenet"
weighted_sampler = True
n_epochs_stop = 5 #how many epochs of no-improve validation loss before early stopping for training
floss_alpha = 0.9 #alpha for focal loss
floss_gamma = 2.4 #gamma for focal loss

#updated by model
phase = "train"
cvphase = 0
trainval = False
num_epochs = 10 #how many epochs to train for
best_epoch = 3 #epoch from training with lowest val loss
best_thresh = 0.0
best_thresh_2tpr = 0.0
best_thresh_2fpr = 0.0

