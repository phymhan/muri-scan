import os
import json
import logging
import shutil
import argparse
import torch
import pdb
from data import MURI, MURIImage
import cv2
import numpy as np
from localbinarypatterns import LocalBinaryPatterns


###############################################################################
# Options | Argument Parser
###############################################################################
class Options():
    def initialize(self, parser):
        parser.add_argument('--mode', type=str, default='train',   help='mode -- train | eval')
        parser.add_argument('--name', type=str, default='TSN',     help='experiment name')

        # Data params
        parser.add_argument('--dataroot',                 default='./data', help='data path')
        parser.add_argument('--imageroot',                default='/media/ligong/Picasso/Datasets/muri/frames-r3')
        parser.add_argument('--labels',         type=str, default='./labels.txt')
        parser.add_argument('--splits',         type=str, default='splits_5.txt', help='text file listing cross-validation splits')
        parser.add_argument('--test_list',      type=str, default='test.txt')
        parser.add_argument('--checkpoint_dir', type=str, default='./experiments/checkpoints', help='directory containing model checkpoints')
        parser.add_argument('--restore_model',  type=str, default='',       help='name of the file that the model is stored')
        
        parser.add_argument('--setting',        type=str, default='openface', help='openface | raw_video')
        parser.add_argument('--num_frames',     type=int, default=180)
        parser.add_argument('--time_stride',    type=int, default=1)

        # Training params
        parser.add_argument('--lr',             type=float,    default=3e-4, help='learning rate')
        parser.add_argument('--batch_size',     type=int,      default=12,   help='batch size')
        parser.add_argument('--num_epochs',     type=int,      default=100,  help='number of epochs')
        parser.add_argument('--num_workers',    type=int,      default=4,    help='number of workers for data loader')
        parser.add_argument('--gpu_ids',        type=str,      default='1',  help='gpu ids: e.g. 0  0,1,2, 0,2 | use '' for CPU')
        parser.add_argument('--train_cont',     type=str2bool, default=False)
        parser.add_argument('--weighted_CE',    type=str2bool, default=True)
        parser.add_argument('--use_gd',         type=str2bool, default=False)
               
        # Model params
        parser.add_argument('--num_classes',    type=int,   default=2,        help='number of classes')
        parser.add_argument('--feature_dim',    type=int,   default=19,       help='dimension of the input fetures (when using the OpenFace setting)')
        parser.add_argument('--kernel_size',    type=int,   default=11,       help='dimension of the kernel size to use for the first Conv')
        parser.add_argument('--embed_dim',      type=int,   default=128,      help='number of filters to use for the first Conv')
        parser.add_argument('--model',          type=str,   default='TCN',    help='name the model to use')
        #parser.add_argument('--dropout',        type=float, default=0.0,      help='dropout rate')
        parser.add_argument('--activation',     type=str,   default='relu',   help='activation function [Relu | LeakyRelu | ELU | Softplus]')
        parser.add_argument('--init_method',    type=str,   default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--dilation',       type=int,   default=3,         help='dilation factor')
        parser.add_argument('--num_blocks',     type=int,   default=3,         help='number of Residual Blocks')


        # Logging params
        parser.add_argument('--tensorboard',     type=str2bool, default=True)
        parser.add_argument('--log_dir',         type=str,      default='./experiments/',    help='directory used for logging')
        parser.add_argument('--tensorboard_dir', type=str,      default='./experiments/TB/', help='directory to save Tensorboard visualizations')

        parser.add_argument('--gradcam_layer',   type=str, default='conv_block1')
        parser.add_argument('--bayesian', type=str2bool, default=False)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--bayesian_T', type=int, default=10)
        return parser

    def get_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        self.opt = self.parser.parse_args()
        self.opt.use_gpu = len(self.opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.print_options(self.opt)
        return self.opt
    
    def print_options(self, opt):
        message = ''
        message += '--------------- Options -----------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        
### ------------------------------------------------------------------------------
# Helper functions for Data handling

def get_splits(splits_file):
    """
    Return:
        splits: dictionary with keys->split_num and values->list with corresponding videos
    """
    splits_num = int(splits_file.split("_")[1].split(".")[0])
    with open(splits_file, 'r') as spl:
        lines = spl.readlines()
    splits = {i : [] for i in range(splits_num)}
    for l in lines:
        splits[int(l.split()[1])].append(l.split()[0])
    # pdb.set_trace()
    return splits


def get_dataset(opt, file_list):

    if opt.setting == 'openface':    
        dataset = MURI(opt.dataroot, file_list, opt.labels, opt.num_frames, opt.time_stride)
    else:
        raise NotImplementedError('Setting [%s] is not implemented.' % opt.setting)
    return dataset

def get_image_dataset(opt, file_list):

    if opt.setting == 'openface':    
        dataset = MURIImage(opt.imageroot, file_list, opt.labels, opt.num_frames, opt.time_stride)
    else:
        raise NotImplementedError('Setting [%s] is not implemented.' % opt.setting)
    return dataset

def get_stats(file_list):
    dd = {"truth": 0, "lie": 0}
    for sample in file_list:
        dd[sample.split("_")[1]] += 1
    return dd    

### ----------------------------------------------------------------------------
# Save and load model

def save_checkpoint(state, is_best, checkpoint):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state:      (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best:    (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model:      (torch.nn.Module) model for which the parameters are loaded
        optimizer:  (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

### ----------------------------------------------------------------------------
# Logging and helper functions

def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter())
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def str2bool(v):
    """
    Borrowed from: https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
    Return: 
        bool(v)
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sample_random_frame_from_video(name):
    vid = cv2.VideoCapture(name)

    _, image = vid.read()
    vid.release()
    return image

    count = 0
    success = 1
    while success:
        success, _ = vid.read()
        count += 1
    vid.release()
    idx = np.random.randint(0, count)
    image = None
    vid = cv2.VideoCapture(name)
    for i in range(idx):
        _, image = vid.read()
    vid.release()
    return image


def get_hists_from_video(filename):
    desc = LocalBinaryPatterns(24, 8)

    vid = cv2.VideoCapture(filename)
    count = 0
    success = 1
    while success:
        success, _ = vid.read()
        count += 1
    vid.release()

    numframe = count
    chunksize = int(numframe / 20.)
    idxs = range(0, numframe, chunksize)[:20]
    hists = []
    idxs_ = []

    for idx in idxs:
        idx = idx + np.random.randint(0, chunksize-2)
        idx = min(idx, numframe)
        idxs_.append(idx)
    
    if len(idxs_) < 20:
        idxs_ = [0] + idxs_

    count = 0
    vid = cv2.VideoCapture(filename)
    while count <= numframe:
        _, image = vid.read()
        if count in idxs_:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hists.append(desc.describe(gray))
        count += 1
    
    hist = np.concatenate(hists)
    vid.release()
    return hist
