import os
import logging
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch.nn import init
import torch.optim as optim
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import softmax

import utils
import models
from models import TCN
from metrics import accuracy

import pdb
from torchsummary import summary
from torch.optim.optimizer import Optimizer, required
global MAGIC_EPS
MAGIC_EPS = 1e-20


class Adam_HD(Optimizer):

    def __init__(self, params, lr=1e-3, lr_lr=.1, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, lr_lr=lr_lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(Adam_HD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam_HD does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['lr'] = group['lr']
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p.data)
                    # For calculating df/dlr
                    state['m_debiased_tm1'] = torch.zeros_like(p.data)
                    state['v_debiased_tm1'] = torch.zeros_like(p.data)

                m, m_debiased_tm1 = state['m'], state['m_debiased_tm1']
                v, v_debiased_tm1 = state['v'], state['v_debiased_tm1']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                m.mul_(beta1).add_(1 - beta1, grad)
                v.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Bias corrections
                m_debiased = m.div(1 - beta1 ** state['step'])
                v_debiased = v.div(1 - beta2 ** state['step'])

                # Update learning rate
                h = grad * (-m_debiased_tm1 / (torch.sqrt(v_debiased_tm1) + group['eps']))
                state['lr'] -= group['lr_lr'] * h.mean()

                p.data.addcdiv_(-state['lr'] * m_debiased, (torch.sqrt(v_debiased) + group['eps']))

                m_debiased_tm1.copy_(m_debiased)
                v_debiased_tm1.copy_(v_debiased)

        return loss



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

def get_model(opt):
    model = None
    if opt.model == 'TCN':
        model = TCN(opt)
    else:
        raise NotImplementedError('Model [{}] is not implemented.'.format(opt.model))

    if opt.mode != 'train' and not opt.bayesian:
        model.eval()
    if opt.use_gpu:
        model.cuda()

    return model

### ----------------------------------------------------------------------------------------

def overfit_one_batch(model, dataloader, optimizer, criterion, metrics, opt):
    
    for epoch in range(opt.num_epochs):    
        model.train()
        train_summ = []
        first_batch = next(iter(dataloader))
        for train_batch, labels_batch in ([first_batch] * 20):
            if opt.use_gpu:
                train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            
            output_batch = model(train_batch)
            loss         = criterion(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss and update using calculated gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # compute metrics for every epoch
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()
            summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
            summary_batch['loss'] = loss.item()
            train_summ.append(summary_batch)
            train_metrics_mean = {metric: np.mean([x[metric] for x in train_summ]) for metric in train_summ[0].keys()}
            train_metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics_mean.items())
            print(train_metrics_string)


### ----------------------------------------------------------------------------------------
def adjust_lr(opt, optimizer, epoch):
    lr = opt.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, criterion, metrics, opt, val_fold):
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
    best_auc       = 0.0 
    best_val_acc   = 0.0
    best_val_acc2  = 0.0 
    best_train_acc = 0.0
    if opt.tensorboard:
        writer = SummaryWriter(opt.tensorboard_dir+opt.name)

    for epoch in range(opt.num_epochs):
        # adjust_lr(opt, optimizer, epoch)        
        ######
        ###   Train model for one epoch
        model.train()
        train_summ = []
        for train_batch, labels_batch in train_dataloader:
            if opt.use_gpu:
                train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            
            output_batch = model(train_batch)
            prob = softmax(output_batch)
            loss = criterion(torch.log(prob + MAGIC_EPS), labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss and update using calculated gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # compute metrics for every epoch
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()
            summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
            summary_batch['loss'] = loss.item()
            train_summ.append(summary_batch)
        
        ######
        ####  Evaluate for one epoch on validation set
        model.eval()
        val_summ  = []
        val_summ2 = []
        preds     = []
        labels    = []
        val_acc2  = 0.0
        for data_batch, labels_batch in val_dataloader:
            if opt.use_gpu:
                data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
            
            prob = 0.
            for t in range(opt.bayesian_T):
                output_batch = model(data_batch)
                prob += 1. / opt.bayesian_T * softmax(output_batch)
            loss         = criterion(torch.log(prob + MAGIC_EPS), labels_batch)
            output_batch = prob.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            labels.append(labels_batch.item())
            preds.append(prob[0][1].item())

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
            summary_batch['loss'] = loss.item()
            val_summ.append(summary_batch)
            val_summ2.append(summary_batch['accuracy'])

        # compute mean of all metrics in summary
        train_metrics_mean = {metric: np.mean([x[metric] for x in train_summ]) for metric in train_summ[0].keys()}
        val_metrics_mean   = {metric: np.mean([x[metric] for x in val_summ]) for metric in val_summ[0].keys()}
        val_acc2 = sum(val_summ2)
        auc      = roc_auc_score(labels, preds)
        # pdb.set_trace()
        
        # Log some info
        if epoch % 10 == 0:
            train_metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics_mean.items())
            val_metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics_mean.items())
            print("Epoch {}/{}".format(epoch+1, opt.num_epochs))
            print("- Train metrics: " + train_metrics_string)
            print("- Eval metrics : " + val_metrics_string)
            print("- AUC: " + str(auc))

        # Log to TensorBoard
        val_acc   = val_metrics_mean['accuracy']
        train_acc = train_metrics_mean['accuracy']
        if opt.tensorboard:
            writer.add_scalar('Loss/Train/f{}'.format(val_fold), train_metrics_mean['loss'], epoch)
            writer.add_scalar('Loss/Val/f{}'.format(val_fold), val_metrics_mean['loss'], epoch)
            writer.add_scalar('Acc/Train/f{}'.format(val_fold), train_acc, epoch)
            writer.add_scalar('Acc/Val/f{}'.format(val_fold), val_acc, epoch)
        
        # save weights
        is_best = val_acc > best_val_acc
        utils.save_checkpoint({'epoch'     : epoch+1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                               is_best     = is_best,
                               checkpoint  = opt.checkpoint_dir)
        
        if is_best:
            print("### Found new best accuracy -> {} -- epoch: {}".format(val_acc, epoch+1))
            best_val_acc = val_acc
              
        best_auc       = max(best_auc, auc)
        best_train_acc = max(best_train_acc, train_acc) 
        best_val_acc2  = max(best_val_acc2, val_acc2)
        if best_val_acc==1.0:
            print("Perfect Accuracy at epoch: {}".format(epoch))
            return best_train_acc, best_val_acc, best_val_acc2, best_auc            

    if opt.tensorboard:
        writer.close()

    return best_train_acc, best_val_acc, best_val_acc2, best_auc



if __name__ == '__main__':

    opt = utils.Options().get_options()
    splits_dict = utils.get_splits(opt.splits)
    # set the random seed for reproducible experiments
    torch.manual_seed(777)
    if opt.use_gpu:
        torch.cuda.manual_seed(777)

    # set the logger
    utils.set_logger(os.path.join(opt.log_dir, 'log.log'))

    # define the model, optimizer and loss function
    model = get_model(opt)
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    optimizer = Adam_HD(model.parameters())
    
    opt.name = opt.name + "_L="+str(opt.kernel_size) +"_e="+str(opt.embed_dim) +"_i="+opt.init_method+\
                "_f="+str(opt.num_frames) +"_ts="+str(opt.time_stride) +"_"+opt.activation
    summary(model, input_size=(opt.feature_dim, opt.num_frames))
    metrics = {'accuracy': accuracy}

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
        
        W = None
        if opt.weighted_CE:
            stats = utils.get_stats(x_train)
            total = stats["truth"] + stats["lie"]
            W     = torch.Tensor([total/stats["truth"], total/stats["lie"]]).cuda()
        # criterion = torch.nn.CrossEntropyLoss(weight=W)
        criterion = torch.nn.NLLLoss(weight=W)

        # Create the input data pipeline
        bs = total_num-batches[i] if opt.use_gd else opt.batch_size
        train_dataloader = DataLoader(utils.get_dataset(opt, x_train), shuffle=True, num_workers=opt.num_workers, batch_size=bs)
        val_dataloader   = DataLoader(utils.get_dataset(opt, x_val),   shuffle=False, num_workers=0, batch_size=1)
        
        # (re)-initialize | (re)-load weights
        if opt.mode == 'train' and not opt.train_cont:    
            init_network(model, init_method=opt.init_method)
        else:
            if opt.train_cont:
                restore_path = os.path.join(opt.checkpoint_dir, opt.restore_model + '.pth.tar')
                print("Restoring parameters from {}".format(restore_path))
                utils.load_checkpoint(restore_path, model, optimizer)

        # Overfit one batch
        # if i==5:
        #     overfit_one_batch(model, train_dataloader, optimizer, criterion, metrics, opt)

        # Train the model
        print("# Val-Fold: {} / {} --- Starting training for {} epoch(s)".format(i+1, folds_num, opt.num_epochs))
        train_acc, val_acc, val_acc2, val_auc = train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, criterion, metrics, opt, i)
        accs_train.append(train_acc)
        accs_val.append(val_acc)
        aucs.append(val_auc)
        
    logging.info("-- {} - {} epochs\n- Train_Acc: {}\n- Val Acc: {}\n- Val AUC: {}".format(opt.name, opt.num_epochs, accs_train, accs_val, aucs) )
    logging.info("### Av. Train_Acc: {} --- Av. Val_Acc: {} --- Av. AUC: {}".format( np.mean(accs_train), max(np.mean(accs_val), val_acc2/total_num), np.mean(aucs)) )
    logging.info("###########")
