import argparse

def create_parser():
    #mobilenet parser
    parser = argparse.ArgumentParser(description="Thyroid CNN Project Pytorch Model Code")
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
    
    return parser


def create_transformer_parser():
    parser = argparse.ArgumentParser(description="Thyroid Transformer Project Pytorch Model Code")
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
    parser.add_argument("--pretrained_dir", type=str, help="folder to get pretrained model weights from in setup_model function")
    parser.add_argument("--modeltype", type=str, default='mobilenet', help="type of cnn model running")
    parser.add_argument("--phase", type=str, default="train", help="train, val, test, etc")
    parser.add_argument("--cvphase", type=int, default=0, help="cross validation fold 0 through 4")
    parser.add_argument("--imgpath", type=str, default="/home/jupyter/containers/images/ALL_IMG_Final.hdf5", help="path to img hdf5 file")  #CHANGE if required
    parser.add_argument("--maskpath", type=str, default="/home/jupyter/containers/images/ALL_MASK_Final.hdf5", help="path to mask hdf5 file")  #CHANGE if required
    parser.add_argument("--labelpath", type=str, default="/home/jupyter/containers/images/frame_lesionId_label_demographics_foldNum.csv", help="path to labels csv file") #CHANGE if required
    parser.add_argument("--features2dpath", type=str, default="/home/jupyter/containers/images/FinalV2_ForTara.csv", help="path to manual 2d features csv file") #CHANGE if required
    parser.add_argument("--project_home_dir", type=str, default="/home/jupyter/", help="home project directory") #CHANGE
    parser.add_argument("--frametype", type=str, default="adjacent", help="adjacent, equalspaced, singleframe (to feed into model)")
    parser.add_argument("--probFunc", type=str, default="Softmax", help="classification function")
    parser.add_argument("--features2d", type=bool, help="whether to concatenate 2d features or not")
    parser.add_argument("--pos_encoding", type=bool, help="whether to add positional encoding in transformer")
    parser.add_argument("--vertconcat", type=bool, help="whether to concatenate 2d features vertically or horizontally")
    parser.add_argument("--weightdecay", type=bool, default=False, help="whether to add weight decay in transformer training")
    parser.add_argument("--transfocalloss", type=bool, default=False, help="whether to use focal loss in transformer training")
    parser.add_argument("--trainval", type=bool, help="whether in training mode with or without validation data (3/5 or 4/5)")
    parser.add_argument("--numpatstrain", type=int, help="Number of patients in transformer train")
    parser.add_argument("--numpatsval", type=int, help="Number of patients in transformer val")
    parser.add_argument("--numpatstest", type=int, help="Number of patients in transformer test")
    
    return parser
