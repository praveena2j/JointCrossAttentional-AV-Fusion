'''Some helper functions for PyTorch, including:
    - progress_bar: progress bar mimic xlua.progress.
    - set_lr : set the learning rate
    - clip_gradient : clip gradient
'''

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
import numpy as np
#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    #for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
    #    sys.stdout.write(' ')

    # Go back to the center of the bar.
    #for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #    sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def accuracy_ccc(label, pred):
    y_true = label[:,0]
    y_pred = pred[:,0]

    cor=np.corrcoef(y_true,y_pred)[0][1]

    mean_true=np.mean(np.array(y_true))
    mean_pred=np.mean(np.array(y_pred))

    var_true=np.var(np.array(y_true))
    var_pred=np.var(np.array(y_pred))

    sd_true=np.std(np.array(y_true))
    sd_pred=np.std(np.array(y_pred))

    numerator = 2*cor*sd_true*sd_pred
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator

def calc_scores (x , y):
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = 1.0 / (len(x)-1) * np.nansum((x-x_mean)*(y-y_mean))
    # covariance = np.nanmean((x - x_mean) * (y - y_mean))
    x_var = 1.0 / (len(x)-1) * np.nansum((x-x_mean)**2)
    y_var = 1.0 / (len(y)-1) * np.nansum((y-y_mean)**2)
    CCC = (2*covariance) / (x_var + y_var + (x_mean-y_mean)**2)
    return CCC

def concordance_correlation_coefficient(prediction, ground_truth):
    """Defines concordance loss for training the model.

    Args:
       prediction: prediction of the model.
       ground_truth: ground truth values.
    Returns:
       The concordance value.
    """
    pred_mean, pred_var = tf.nn.moments(prediction, (0,))
    gt_mean, gt_var = tf.nn.moments(ground_truth, (0,))

    mean_cent_prod = tf.reduce_mean((prediction - pred_mean) * (ground_truth - gt_mean))
    return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr
    return lr 

def OptimizePostProcessing ( predictions, labels):
    bestScore = -1
    bestWindow = 0
    for w in range (1,200,2):
        newPred = signal.medfilt (predictions, w)
        ccc = accuracy_ccc (labels, newPred)
        if ccc> bestScore:
            bestScore = ccc
            bestWindow = w
    print (bestScore, bestWindow)
    return signal.medfilt (predictions, bestWindow)


def MinmaxNormalize(data):
    maxVal = np.nanmax(data)
    minVal = np.nanmin(data)
    return [(x-minVal)/(maxVal-minVal) for x in data]

def MeanNormalization (ref, pred):
    mean_ref = np.mean (ref)
    mean_pred = np.mean (pred)
    bias = mean_ref - mean_pred
    return pred + bias

def GetStdRatio(labels, pred):
    label_sd = np.std(labels)
    pred_sd = np.std(pred)
    sd = label_sd / pred_sd
    return sd

def ApplyScaling (pred, ratio):
    return [x*ratio for x in pred]

def MedianFilter (data, windowLength):
    return signal.medfilt (data, windowLength)

def Normalize (pred, devLabels):
    # std prediction normalization
    #predTrain = reg.predict (trainData)

    #devLabels = devLabels
    #pred = pred

    #filt_devLabels = MedianFilter (devLabels, 3)
    #filt_pred = MedianFilter (pred, 3)

    if (np.nanmax(devLabels) - np.nanmin(devLabels)) == 0:
        print(devLabels)
        tar = devLabels
    else:
        tar = MinmaxNormalize(devLabels)

    if (np.nanmax(pred) - np.nanmin(pred)) == 0:
        print(pred)
    else:        
        pred = MinmaxNormalize(pred)

    # pred = MeanNormalization (devLabels, pred) 
    # ratio = GetStdRatio (devLabels, pred)
    # pred = ApplyScaling (pred, ratio)

    # Mean bias normalization
    pred = MeanNormalization(tar, pred)
    return pred, tar

def save_checkpoint(state, filename):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    scale = 0.457305051927326
    step  = 10
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #maxk = max(topk)
    batch_size = target.size(0)

    #print(maxk)
    #print(output)
    #_, pred = output.topk(maxk, 1, True, True)

    #pred = pred.t()
    correct = output.eq(target)

    correct_ar = correct[:,0]
    correct_va = correct[:,1]

    ar_res = []
    ar_corrected = correct_ar.view(-1).float().sum(0)
    ar_res.append(ar_corrected.mul_(100.0 / batch_size))
    
    va_res = []
    va_corrected = correct_va.view(-1).float().sum(0)
    va_res.append(va_corrected.mul_(100.0 / batch_size))
    return ar_res, va_res

def accuracy_reg(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    
    pred = pred.t()
    target = target.long()

    correct = pred.eq(target.expand_as(pred))

    res = []
    corrected = correct.view(-1).float().sum(0)
    res.append(corrected.mul_(100.0 / batch_size))
    return res

def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
    print('There are {} images..'.format(len(img_list)))
    return img_list

def save_feature(save_path, img_name, features):
    img_path = os.path.join(save_path, img_name)
    img_dir  = os.path.dirname(img_path) + '/';
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fname = os.path.splitext(img_path)[0]
    fname = fname + '.feat'
    fid   = open(fname, 'wb')
    fid.write(features)
    fid.close()

