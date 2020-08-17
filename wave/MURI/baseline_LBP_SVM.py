import os
import logging
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

import torch
from torch.nn import init
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import softmax

import utils
import models

import pdb
from torchsummary import summary
from torch.optim.optimizer import Optimizer, required
from localbinarypatterns import LocalBinaryPatterns
from sklearn.metrics import accuracy_score


##### ---------------------------------------------------------------------

def init_network(net, init_method='xavier', gain=1):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_method == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_method == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_method == 'kaiming':
                # use only when using Relu or LRelu
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_method == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [{}] is not implemented'.format(init_method))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.001)
            init.constant_(m.bias.data, 0.0)

    print('Initialize network with [{}]'.format(init_method))
    net.apply(init_func)


### ----------------------------------------------------------------------------------------
def adjust_lr(opt, optimizer, epoch):
    lr = opt.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_and_evaluate(train_dataloader, val_dataloader, opt, val_fold):
    """
    Train the model and evaluate every epoch.

    Args:
        model:            (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader:   (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer:        (torch.optim) optimizer for parameters of model
        criterion:        a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics:          (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        opt:              Options
    """
    data = []
    labels = []
    
    # loop over the training images
    for hist, label in train_dataloader:
        hist = hist.detach().cpu().numpy().squeeze()
        label = 2 * label - 1
        labels.append(label.item())
        data.append(hist)

    # train a Linear SVM on the data
    model = LinearSVC(C=100.0, random_state=42)
    # pdb.set_trace()
    model.fit(data, labels)

    preds     = []
    labels    = []
    # test on training images
    for hist, label in train_dataloader:
        hist = hist.detach().cpu().numpy().squeeze()
        label = 2 * label - 1

        prediction = model.predict(hist.reshape(1, -1))
        labels.append(label.item())
        preds.append(1 if prediction > 0 else -1)

    # compute all metrics
    labels, preds = (np.array(labels)+1)/2, (np.array(preds)+1)/2
    train_acc = accuracy_score(preds, labels)
    train_auc = roc_auc_score(labels, preds)
    
    preds     = []
    labels    = []
    # loop over the testing images
    for hist, label in val_dataloader:
        hist = hist.detach().cpu().numpy().squeeze()
        label = 2 * label - 1
        prediction = model.predict(hist.reshape(1, -1))
        labels.append(label.item())
        preds.append(1 if prediction > 0 else -1)

    # compute all metrics
    labels, preds = (np.array(labels)+1)/2, (np.array(preds)+1)/2
    val_acc = accuracy_score(preds, labels)
    val_auc = roc_auc_score(labels, preds)

    return train_acc, val_acc, train_auc, val_auc



if __name__ == '__main__':

    opt = utils.Options().get_options()
    splits_dict = utils.get_splits(opt.splits)
    # set the random seed for reproducible experiments
    torch.manual_seed(777)
    if opt.use_gpu:
        torch.cuda.manual_seed(777)
  
    opt.name = opt.name + "_L="+str(opt.kernel_size) +"_e="+str(opt.embed_dim) +"_i="+opt.init_method+\
                "_f="+str(opt.num_frames) +"_ts="+str(opt.time_stride) +"_"+opt.activation

    # run k iterations of K-fold cross-validation
    accs_train = []
    accs_val   = []
    aucs = []
    batches = [len(splits_dict[i]) for i in splits_dict.keys()]
    total_num = sum(batches)
    folds_num = len(splits_dict.keys())
    for i in range(folds_num):
        x_val = splits_dict[i].copy()
        x_train = []

        for j in range(folds_num):
            if j!=i:
                x_train = x_train + splits_dict[j]
        
        # Create the input data pipeline
        train_dataloader = DataLoader(utils.get_image_dataset(opt, x_train), shuffle=True, num_workers=0, batch_size=1)
        val_dataloader   = DataLoader(utils.get_image_dataset(opt, x_val),   shuffle=False, num_workers=0, batch_size=1)
        
        # Train the model
        print("# Val-Fold: {} / {} --- Starting training for {} epoch(s)".format(i+1, folds_num, opt.num_epochs))
        train_acc, val_acc, train_auc, val_auc = train_and_evaluate(train_dataloader, val_dataloader, opt, i)
        accs_train.append(train_acc)
        accs_val.append(val_acc)
        aucs.append(val_auc)
    
    logging.info("-- {} - {} epochs\n- Train_Acc: {}\n- Val Acc: {}\n- Val AUC: {}".format(opt.name, opt.num_epochs, accs_train, accs_val, aucs) )
    logging.info("### Av. Train_Acc: {} --- Av. Val_Acc: {} --- Av. Val_AUC: {}".format(np.mean(accs_train), np.mean(accs_val), np.mean(accs_val), np.mean(aucs)) )
    print("### Av. Train_Acc: {} --- Av. Val_Acc: {} --- Av. Val_AUC: {}".format(np.mean(accs_train), np.mean(accs_val), np.mean(accs_val), np.mean(aucs)) )
    logging.info("###########")
