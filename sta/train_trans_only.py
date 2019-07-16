import torch
#from util.parse_data import *
#from util.data_loader import *
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from datetime import datetime
import time
from tensorboardX import SummaryWriter
import argparse
import os
import data

from transformer_only import *

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=32)  # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')  # 1000

parser.add_argument('--patiences', default=100, type=int,
                    help='number of epochs to tolerate the no improvement of val_loss')  # 1000

parser.add_argument('--class_num', default=2, type=int,
                    help='number of class')  # 1000

parser.add_argument('--test_subject_id', type=int,
                    help='id of test subject')  # 1000

parser.add_argument('--random_seed', type=int,
                    help='random seed for init parameters')  # 1000

parser.add_argument('--data_cfg', type=int,
                    help='0 for 14 class, 8 for 28')  # 1000


parser.add_argument('--att_loss_W', default=1, type=float,
                    help='0 for 14 class, 8 for 28')  # 1000

parser.add_argument('--use_graph', type=int,
                    help='wether add graph constrains to graph')  # 1000

parser.add_argument('--dp_rate', type=float,
                    help='dropout rate')  # 1000

parser.add_argument('--lr_decay_factor', type=float, default=0.1,
                    help='dropout rate')  # 1000


def init_data_loader(test_subject_id, args, cfg):
    
    train_dataset = data.MURI_Dataset('/home/lh599/Research/MURI/openface/clips-r3', 'filelist_0.txt', 'labels.txt', 32)
    test_dataset = data.MURI_Dataset('/home/lh599/Research/MURI/openface/clips-r3', 'filelist_0.txt', 'labels.txt', 32)
    # train_data, test_data = get_train_test_data(test_subject_id, cfg)

    # train_dataset = DHS_Dataset(train_data, use_data_aug = True, time_len = 8, sample_strategy = "equi_T")

    # test_dataset = DHS_Dataset(test_data, use_data_aug = False, time_len = 8, sample_strategy = "equi_T")

    print("train data num: ",len(train_dataset))
    print("test data num: ",len(test_dataset))

    print("batch size:", args.batch_size)
    print("workers:", args.workers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader

def init_model(data_cfg):
    class_num = 2

    model = Trans_only(class_num, bool(args.use_graph), args.dp_rate)
    model = torch.nn.DataParallel(model).cuda()

    return model


def model_foreward(sample_batched,model,criterion):

    # def compute_att_loss(model):
    #     loss = 0
    #     for l_id in range(1, 11, 2):
    #         att_score = model.module.trans_encoder.gcn_network[l_id].layers[
    #             0].self_attn.attn  # [batch, head_num, row, column]
    #         #print((1. - att_score.sum(dim=2).shape))
    #         att_regl = ((1. - att_score.sum(dim=2)) ** 2).mean()
    #         loss += att_regl
    #     loss /= 6
    #     return loss

    data = sample_batched["skeleton"].float()
    #print(data.shape)
    #print(data.shape)

    label = sample_batched["label"]
    label = label.type(torch.LongTensor)
    label = label.cuda()
    label = torch.autograd.Variable(label, requires_grad=False)

    #np.savetxt("foo.csv", adj[0].cpu().data.numpy(), delimiter=",")

    score = model(data)
    #att_loss = compute_att_loss(model)
    att_loss = 0

    loss = criterion(score,label)

    acc = get_acc(score, label)

    return score,loss,att_loss, acc



def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# def adjust_learning_rate(optimizer, no_improve_epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.learning_rate * (0.1 ** (no_improve_epoch // 20))
#     print("[lr: {}, epoch {}]".format(lr,epoch))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# def adjust_learning_rate(optimizer, shrink_factor):
# 	print("\nDECAYING learning rate.")
# 	for param_group in optimizer.param_groups:
# 		param_group['lr'] = param_group['lr'] * shrink_factor
# 	print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr']))
# 	return optimizer.param_groups[0]['lr']


torch.manual_seed(999)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    print("\nhyperparamter......")
    args = parser.parse_args()
    print(args)

    model_fold = "/home/lh599/Active/muri-scan/model/{}_dp-{}_lr-{}_g-{}_dc-{}/".format("trans_only", args.dp_rate, args.learning_rate, bool(args.use_graph), args.data_cfg)
    try:
        os.mkdir(model_fold)
    except:
        pass



    #........get data loader
    #test_subject_id = 1

    test_subject_id = args.test_subject_id

    print("test_subject_id: ", test_subject_id)
    train_loader, val_loader = init_data_loader(test_subject_id, args, cfg=args.data_cfg)


    #.........inital model
    print("\ninit model.............")
    model = init_model(args.data_cfg)
    model_solver = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    #........set loss
    criterion = torch.nn.CrossEntropyLoss()

    #.........tensorboard
    now = datetime.now()
    log_path = "/home/lh599/Active/muri-scan/log/{}_dp-{}_lr-{}_g-{}_dc-{}/{}/sub_id-{}/".format("trans_only", args.dp_rate, args.learning_rate, bool(args.use_graph), args.data_cfg, now.strftime("%Y%m%d-%H%M%S"), test_subject_id)
    writer = SummaryWriter(log_path)

    #..........parameters use to show traing status
    train_data_num = 2800 / 20 * 19
    test_data_num = 2800 - train_data_num
    iter_per_epoch = int(train_data_num / args.batch_size)
    test_epoch = int(train_data_num / args.batch_size)


    max_acc = 0
    no_improve_epoch = 0
    decay_count = 0
    n_iter = 0


    for epoch in range(args.epochs):
        print("\ntraining.............")
        model.train()
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        train_att_loss = 0
        for i, sample_batched in enumerate(train_loader):
            n_iter += 1

            if i + 1 > iter_per_epoch:
                continue
            score,loss,att_loss,acc = model_foreward(sample_batched, model, criterion)
            #print(score)
            #print(loss)
            #dd

            #back_ward
            model.zero_grad()
            #print(att_loss)
            #train_att_loss += att_loss

            loss.backward()
            model_solver.step()


            train_acc += acc
            train_loss += loss

            #print(i)



        train_acc /= float(i + 1)
        train_loss /= float(i + 1)

        writer.add_scalar('Train/Loss', train_loss.data, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)

        print("*** Subject_ID: [%2d]  Epoch: [%2d] time: %4.4f, "
              "cls_loss: %.4f  train_ACC: %.6f ***"
              % (test_subject_id, epoch + 1,  time.time() - start_time,
                 train_loss.data, train_acc))
        start_time = time.time()

        #adjust_learning_rate(model_solver, epoch + 1, args)
        #print(print(model.module.encoder.gcn_network[0].edg_weight))

        #do evaluation
        with torch.no_grad():
            val_loss = 0
            acc_sum = 0
            model.eval()
            val_att_loss = 0
            for i, sample_batched in enumerate(val_loader):
                label = sample_batched["label"]
                score, loss, att_loss, acc = model_foreward(sample_batched, model, criterion)
                val_loss += loss
                val_att_loss += att_loss

                if i == 0:
                    score_list = score
                    label_list = label
                else:
                    score_list = torch.cat((score_list, score), 0)
                    label_list = torch.cat((label_list, label), 0)


            val_loss = val_loss / float(i + 1)
            val_att_loss /= float(i + 1)
            #print(score_list.shape)
            #print(label_list.shape)
            val_cc = get_acc(score_list,label_list)

            writer.add_scalar('Test/Loss', val_loss.data, epoch)
            writer.add_scalar('Test/Accuracy', val_cc, epoch)

            print("*** Subject_ID: [%2d]  Epoch: [%2d], "
                  "val_loss: %.6f,"
                  "val_ACC: %.6f ***"
                  % (test_subject_id, epoch + 1, val_loss, val_cc))

            #save model
            if val_cc > max_acc:
                max_acc = val_cc
                no_improve_epoch = 0
                decay_count = 0
                val_cc = round(val_cc, 10)

                outf_path = model_fold + "sub-{}".format(test_subject_id)

                if not os.path.exists(outf_path):
                    os.mkdir(outf_path)
                torch.save(model.state_dict(),
                           '{}/epoch_{}_acc_{}.pth'.format(outf_path, epoch + 1, val_cc))
                print("performance improve, saved the new model......best acc: {}".format(max_acc))
            else:
                no_improve_epoch += 1
                decay_count += 1
                print("no_improve_epoch: {} best acc {}".format(no_improve_epoch,max_acc))
                print("decay_count:",decay_count)

            # if decay_count >= 20:
            #     decay_count = 0
            #     adjust_learning_rate(model_solver, args.lr_decay_factor)
            if no_improve_epoch > args.patiences:
                print("stop training....")
                break
