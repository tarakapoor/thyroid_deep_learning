import os
import torch

def setup_model(net, phase, cvphase, model_dir, pretrained_dir, project_home_dir, best_epoch):
    print("setting up model")
    
    model_path = project_home_dir + 'model/' + model_dir + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print('Make the folder', model_path, 'to save model---------\n')
    if phase == 'test':
        load_path = project_home_dir + 'model/' + pretrained_dir + '/' + pretrained_dir +'_cv%s_epoch%s.pth' % (cvphase, best_epoch)
        
        print('loading the model from %s' % load_path)
        
        #for cnn
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    save_path = model_path
    save_file_name = save_path + model_dir + '_cv%s_epoch%s.pth' % (cvphase, epoch)
    print("model save path:", save_file_name)
    torch.save(net.state_dict(), save_file_name)
    
print("done with save functions!")
