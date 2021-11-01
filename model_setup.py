import os
import torch


def setup_model(net, phase, cvphase, model_dir, pretrained_dir, project_home_dir, best_epoch):
    """Set up model.
    Either create model directories to save to (train phase), or load pretrained model weights state dict from pretrained directory (test phase).
    Keyword arguments:
    net -- model with weights to save
    phase -- test or something else (used to determine whether to use pretrained weights)
    cvphase -- cross validation fold (0 to 4), used in model save directory/path name
    model_dir -- where model is saved if not in test phase
    pretrained_dir -- where to load model pretrained weights from if in test phase
    best_epoch -- which epoch weights to load if in test phase
    """

    print("setting up model")
    
    model_path = project_home_dir + 'model/' + model_dir + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print('Make the folder', model_path, 'to save model---------\n')
    
    if phase == 'test':
        load_path = project_home_dir + 'model/' + pretrained_dir + '/' + pretrained_dir +'_cv%s_epoch%s.pth' % (cvphase, best_epoch)
        print('loading the model from %s' % load_path)
        
        #for cnn
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #use gpu if possible
        pretrained_state_dict = torch.load(load_path, map_location=str(device))
        
        try:
            net.load_state_dict(pretrained_state_dict)
            print("successfully loaded pretrained state dict")
            #load the weights from the trained model to use in test phase?
        except:
            try:
                model_dict = net.state_dict()
                pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
                net.load_state_dict(pretrained_state_dict)
            except:
                print(
                    'Pretrained network has fewer layers; The following are not initialized:')
                for k, v in pretrained_state_dict.items():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v

                not_initialized = set()
                for k, v in model_dict.items():
                    if k not in pretrained_state_dict or v.size() != pretrained_state_dict[k].size():
                        not_initialized.add(k.split('.')[0])

                print(sorted(not_initialized))
                net.load_state_dict(model_dict)


    
def save_networks(net, epoch, cvphase, model_path, model_dir):
    """Save model weights to model path.
    Keyword arguments:
    net -- model with weights to save
    epoch -- epoch number used in model save directory/path name
    cvphase -- cross validation fold (0 to 4), used in model save directory/path name
    model_path -- used with model dir for save path
    model_dir -- folder after model path where model is saved
    """
    save_file_name = model_path + model_dir + '_cv%s_epoch%s.pth' % (cvphase, epoch)
    print("model save path:", save_file_name)
    torch.save(net.state_dict(), save_file_name)
    
print("done with save functions!")