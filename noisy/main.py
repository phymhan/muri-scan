import os
import argparse
import random
import functools
import math
import numpy as np
from scipy import stats
import scipy.io as sio
import torch
from torch import optim
from models import BaseClipModel, BaseVideoModel
from utils import init_net, make_dir, str2bool, set_logger
from data import ClipDataset, VideoDataset
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value
import torch.nn.functional as F

import pdb

global MAGIC_EPS
MAGIC_EPS = 1e-32


###############################################################################
# Options | Argument Parser
###############################################################################
class Options():
    def initialize(self, parser):
        parser.add_argument('--mode', type=str, default='train', help='train | test')
        parser.add_argument('--name', type=str, default='exp', help='experiment name')
        parser.add_argument('--dataroot', default='../data/npz', help='path to images')
        parser.add_argument('--sourcefile', type=str, default='../sourcefiles/train.txt', help='text file listing images')
        parser.add_argument('--sourcefile_val', type=str, default='../sourcefiles/val.txt')
        parser.add_argument('--labelfile', type=str, default='../sourcefiles/labels.txt')
        parser.add_argument('--pretrained_model_path', type=str, default='', help='path to pretrained models')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints')
        parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
        parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=100, help='batch size')
        parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
        parser.add_argument('--which_model', type=str, default='base', help='which model')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout p')
        parser.add_argument('--print_freq', type=int, default=50, help='print loss every print_freq iterations')
        parser.add_argument('--continue_train', type=str2bool, default=False)
        parser.add_argument('--epoch_count', type=int, default=1, help='starting epoch')
        parser.add_argument('--tensorboard', type=str2bool, default=True)
        parser.add_argument('--time_len', type=int, default=24)
        parser.add_argument('--time_step', type=int, default=5)
        parser.add_argument('--use_gru', type=str2bool, default=False)
        parser.add_argument('--feature_dim', type=int, default=31)
        parser.add_argument('--embedding_dim', type=int, default=128)
        parser.add_argument('--gru_hidden_dim', type=int, default=32)
        parser.add_argument('--gru_out_dim', type=int, default=8)
        parser.add_argument('--ce_weight', nargs='+', type=float, default=[], help='weights for CE')
        parser.add_argument('--setting', type=str, default='clip', help='clip or video')
        parser.add_argument('--noisy', type=str2bool, default=False)
        parser.add_argument('--lambda_trace', type=float, default=0.001)
        return parser

    def get_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        self.opt = self.parser.parse_args()
        self.opt.use_gpu = len(self.opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.opt.isTrain = self.opt.mode == 'train'
        # weight
        if self.opt.ce_weight:
            assert(len(self.opt.ce_weight) == self.opt.num_classes)
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

        # save to the disk
        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


###############################################################################
# Dataset and Dataloader
###############################################################################
# import from data


###############################################################################
# Networks and Models
###############################################################################
# import from models


###############################################################################
# Helper Functions | Utilities
###############################################################################
def get_prediction(score):
    """
    Majority voting
    """
    batch_size = score.size(0)
    score_cpu = score.detach().cpu().numpy()
    pred = stats.mode(score_cpu.argmax(axis=1).reshape(batch_size, -1), axis=1)
    return pred[0].reshape(batch_size)


###############################################################################
# Main Routines
###############################################################################
def get_dataset(opt, mode='train'):
    if opt.setting == 'clip':
        if mode == 'train':
            dataset = ClipDataset(opt.dataroot, opt.sourcefile, opt.labelfile, opt.time_len, opt.time_step)
        else:
            dataset = ClipDataset(opt.dataroot, opt.sourcefile_val, opt.labelfile, opt.time_len, opt.time_step)
    elif opt.setting == 'video':
        if mode == 'train':
            dataset = VideoDataset(opt.dataroot, opt.sourcefile, opt.labelfile, opt.time_len, opt.time_step)
        else:
            dataset = VideoDataset(opt.dataroot, opt.sourcefile_val, opt.labelfile, opt.time_len, opt.time_step)
    else:
        raise NotImplementedError('Setting [%s] is not implemented.' % opt.setting)
    return dataset


def get_model(opt):
    # define model
    net = None
    if opt.setting == 'clip':
        if opt.which_model == 'base':
            net = BaseClipModel(num_classes=opt.num_classes, use_gru=opt.use_gru, feature_dim=opt.feature_dim, embedding_dim=opt.embedding_dim,
                                gru_hidden_dim=opt.gru_hidden_dim, gru_out_dim=opt.gru_out_dim, dropout=opt.dropout, noisy=opt.noisy)
        else:
            raise NotImplementedError('Model [%s] is not implemented.' % opt.which_model)
    elif opt.setting == 'video':
        if opt.which_model == 'base':
            net = BaseVideoModel(num_classes=opt.num_classes, use_gru=opt.use_gru, feature_dim=opt.feature_dim, embedding_dim=opt.embedding_dim,
                                 gru_hidden_dim=opt.gru_hidden_dim, gru_out_dim=opt.gru_out_dim, dropout=opt.dropout, noisy=opt.noisy)
        else:
            raise NotImplementedError('Model [%s] is not implemented.' % opt.which_model)
    else:
        raise NotImplementedError('Setting [%s] is not implemented.' % opt.setting)
    
    # initialize | load weights
    if opt.mode == 'train' and not opt.continue_train:
        init_net(net, init_type=opt.init_type)
        if opt.noisy:
            # init transition matrix as identity
            net.transition.weight.data.copy_(torch.eye(opt.num_classes))
        if opt.pretrained_model_path:
            if isinstance(net, torch.nn.DataParallel):
                net.module.load_pretrained(opt.pretrained_model_path)
            else:
                net.load_pretrained(opt.pretrained_model_path)
    else:
        net.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, opt.name, '{}_net.pth'.format(opt.which_epoch))))
    
    if opt.mode != 'train':
        net.eval()
    
    if opt.use_gpu:
        net.cuda()
    return net


# Routines for training
def train(opt, net, dataloader):
    if len(opt.ce_weight):
        ce_weight = torch.Tensor(opt.ce_weight).cuda() if opt.use_gpu else torch.Tensor(opt.ce_weight)
        # criterion = torch.nn.CrossEntropyLoss(weight=ce_weight)
        criterion = torch.nn.NLLLoss(weight=ce_weight)
    else:
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.NLLLoss()
    opt.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    make_dir(opt.save_dir)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    dataset_size, dataset_size_val = opt.dataset_size, opt.dataset_size_val
    total_iter = 0
    num_iter_per_epoch = math.ceil(dataset_size / opt.batch_size)
    opt.display_val_acc = not not dataloader_val

    if opt.tensorboard:
        configure(opt.save_dir)
    logger = set_logger(opt.save_dir, opt.name+'.log')
    logger.info('config %s', opt)

    for epoch in range(opt.epoch_count, opt.num_epochs+opt.epoch_count):
        ###############################################################################
        # Train one epoch
        ###############################################################################
        epoch_iter = 0
        pred_train = []
        target_train = []

        for i, data in enumerate(dataloader, 0):
            x, y = data
            if opt.use_gpu:
                x, y = x.cuda(), y.cuda()
            epoch_iter += 1
            total_iter += 1
            optimizer.zero_grad()
            y_pred = net(x)
            y_pred = F.softmax(y_pred)
            if opt.noisy:
                mat = F.relu(net.transition.weight)
                mat = mat / (torch.sum(mat, dim=0, keepdim=True) + MAGIC_EPS)
                y_pred = torch.mm(y_pred, mat)
                logsoftmax = torch.log(y_pred + MAGIC_EPS)
                trace = torch.trace(net.transition.weight)
                loss = criterion(logsoftmax, y) + opt.lambda_trace * trace
            else:
                logsoftmax = torch.log(y_pred + MAGIC_EPS)
                loss = criterion(logsoftmax, y)
            # get predictions
            pred_train.append(get_prediction(y_pred))
            target_train.append(y.cpu().numpy())
            loss.backward()
            optimizer.step()
            losses = {'loss': loss.item()}
            if total_iter % opt.print_freq == 0:
                logger.info(f'[train] epoch {epoch:02d}, iter {epoch_iter:03d}/{num_iter_per_epoch}, loss {loss.item():.4f}')
                if opt.tensorboard:
                    for k in losses:
                        log_value(f'train/{k}', losses[k], total_iter)

        # evaluate training
        err_train = np.count_nonzero(np.concatenate(pred_train) - np.concatenate(target_train)) / dataset_size
        logger.info(f'[train] epoch {epoch:02d}, acc {(1 - err_train) * 100:.2f}%')
        if opt.tensorboard:
            log_value(f'train/acc', 1 - err_train, epoch)

        # evaluate val
        if opt.display_val_acc:
            with torch.no_grad():
                pred_val = []
                target_val = []
                for i, data in enumerate(dataloader_val, 0):
                    x, y = data
                    if opt.use_gpu:
                        x, y = x.cuda(), y.cuda()
                    y_pred = net(x)
                    y_pred = F.softmax(y_pred)
                    if opt.noisy:
                        mat = F.relu(net.transition.weight)
                        mat = mat / (torch.sum(mat, dim=0, keepdim=True) + MAGIC_EPS)
                        y_pred = torch.mm(y_pred, mat)
                    pred_val.append(get_prediction(y_pred))
                    target_val.append(y.cpu().numpy())
            err_val = np.count_nonzero(np.concatenate(pred_val) - np.concatenate(target_val)) / dataset_size_val
            logger.info(f'[val] epoch {epoch:02d}, acc {(1 - err_val) * 100:.2f}%')
            if opt.noisy:
                print('--> transition matrix')
                print(net.transition.weight.data)
            if opt.tensorboard:
                log_value(f'val/acc', 1 - err_val, epoch)

        torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, 'latest_net.pth'))
        if opt.use_gpu:
            net.cuda()
        if epoch % opt.save_epoch_freq == 0:
            torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, '{}_net.pth'.format(epoch)))
            if opt.use_gpu:
                net.cuda()


###############################################################################
# main()
###############################################################################
# TODO: set random seed

if __name__=='__main__':
    opt = Options().get_options()

    # get model
    net = get_model(opt)

    if opt.mode == 'train':
        # get dataloader
        dataset = get_dataset(opt, 'train')
        dataloader = DataLoader(dataset, shuffle=True, num_workers=opt.num_workers, batch_size=opt.batch_size)
        opt.dataset_size = len(dataset)
        # val dataset
        if opt.sourcefile_val:
            dataset_val = get_dataset(opt, 'val')
            dataloader_val = DataLoader(dataset_val, shuffle=True, num_workers=0, batch_size=1)
            opt.dataset_size_val = len(dataset_val)
        else:
            dataloader_val = None
            opt.dataset_size_val = 0
        print('dataset size = %d' % len(dataset))
        # train
        train(opt, net, dataloader)
    else:
        raise NotImplementedError('Mode [%s] is not implemented.' % opt.mode)
