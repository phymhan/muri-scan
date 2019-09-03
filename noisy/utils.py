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

def get_X_y(dataset):
    X = []
    y = []
    for i,(a, b) in enumerate(dataset):
        X.append(a)
        y.append(b)
    return X, y

def k_folds(dataset, k_splits=3):
    '''
    Generates K-folds for cross-validation

    Args:
        k_splits: Number of k_folds,
        dataset: np.zeros()
    '''
    dataset_size = len(dataset)
    indices = np.arange(dataset_size).astype(int)
    for valid_idx in get_indices(dataset_size, k_splits):
        train_idx = np.setdiff1d(indices, valid_idx)
        yield np.take(dataset, train_idx), np.take(dataset, valid_idx)

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


def stratified_k_folds(dataset, k_splits=3):
    '''
    Generates K-folds for cross-validation

    Args:
        k_splits: Number of k_folds,
        dataset_size: Size of the dataset : For even distribution in k_folds
    '''
    X, y = get_X_y(dataset)

    dataset_size = len(X)
    indices = np.arange(dataset_size).astype(int)
    for valid_idx in stratified_get_indices(X, y, k_splits):
        train_idx = np.setdiff1d(indices, valid_idx)
        yield train_idx, valid_idx


def stratified_get_indices(X, y, k_splits=3):
    '''
    Get indices of the dictionary for each split

    Args:
        k_splits : Number of k_folds,
        dataset_size: Size of the dataset : For even distribution in k_folds
    '''

    y = np.asarray(y)
    n_samples = y.shape[0]
    unique_y, y_inversed = np.unique(y, return_inverse=True)
    y_counts = np.bincount(y_inversed)
    min_groups = np.min(y_counts)
    if np.all(k_splits > y_counts):
        raise ValueError("k_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (k_splits))
    if k_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is too few. The minimum"
                           " number of members in any class cannot"
                           " be less than k_splits=%d."
                           % (min_groups, k_splits)), Warning)

    # pre-assign each sample to a test fold index using individual KFold
    # splitting strategies for each class so as to respect the balance of
    # classes
    # NOTE: Passing the data corresponding to ith class say X[y==class_i]
    # will break when the data is not 100% stratifiable for all classes.
    # So we pass np.zeroes(max(c, k_splits)) as data to the KFold

    per_cls_cvs = [
            k_folds(np.zeros(max(count, k_splits), dtype=np.int), k_splits)
            for count in y_counts]

    test_folds = np.zeros(n_samples, dtype=np.int)
    for test_fold_indices, per_cls_splits in enumerate(zip(*per_cls_cvs)):
            for cls, (_, test_split) in zip(unique_y, per_cls_splits):
                cls_test_folds = test_folds[y == cls]
                # the test split can be too big because we used
                # KFold(...).split(X[:max(c, n_splits)]) when data is not 100%
                # stratifiable for all the classes
                # (we use a warning instead of raising an exception)
                # If this is the case, let's trim it:
                test_split = test_split[test_split < len(cls_test_folds)]
                cls_test_folds[test_split] = test_fold_indices
                test_folds[y == cls] = cls_test_folds

    for i in range(k_splits):
        yield test_folds == i




    # k_partitions = np.ones(k_splits) * int(dataset_size/k_splits)
    # k_partitions[0: (dataset_size % k_splits)] += 1
    # indices = np.arange(dataset_size).astype(int)
    # current = 0
    # for each_partition in k_partitions:
    #     start = current
    #     stop = current + each_partition
    #     current = stop
    #     yield (indices[int(start):int(stop)])
