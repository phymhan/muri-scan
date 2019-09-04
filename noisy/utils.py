import argparse
import torch
from torch.nn import init
import os
import logging
import numpy as np

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



def str2bool(v):
    """
    borrowed from:
    https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
    :param v:
    :return: bool(v)
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_logger(output_dir=None, log_file=None):
    head = '%(asctime)-15s Host %(message)s'
    logger_level = logging.INFO
    if all((output_dir, log_file)) and len(log_file) > 0:
        logger = logging.getLogger()
        log_path = os.path.join(output_dir, log_file)
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logger_level)
    else:
        logging.basicConfig(level=logger_level, format=head)
        logger = logging.getLogger()
    return logger


def k_folds(dataset_size, k_splits=3):
    '''
    Generates K-folds for cross-validation

    Args:
        k_splits: Number of k_folds,
        dataset_size: Size of the dataset : For even distribution in k_folds
    '''

    indices = np.arange(dataset_size).astype(int)
    for valid_idx in get_indices(dataset_size, k_splits):
        train_idx = np.setdiff1d(indices, valid_idx)
        yield train_idx, valid_idx


def get_indices(dataset_size, k_splits=3):
    '''
    Get indices of the dictionary for each split

    Args:
        k_splits : Number of k_folds,
        dataset_size: Size of the dataset : For even distribution in k_folds
    '''
    k_partitions = np.ones(k_splits) * int(dataset_size/k_splits)
    k_partitions[0: (dataset_size % k_splits)] += 1
    indices = np.arange(dataset_size).astype(int)
    current = 0
    for each_partition in k_partitions:
        start = current
        stop = current + each_partition
        current = stop
        yield (indices[int(start):int(stop)])
