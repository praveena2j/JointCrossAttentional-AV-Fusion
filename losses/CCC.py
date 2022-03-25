import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import sys

class CCC(nn.Module):

    def __init__(self):
        super().__init__()
        #self.mean = torch.mean
        #self.var = torch.var
        #self.sum = torch.sum
        #self.sqrt = torch.sqrt
        #self.std = torch.std

    def forward(self, prediction, ground_truth):

        prediction = prediction.view(-1)
        #ground_truth = ground_truth.squeeze()
        ground_truth = ground_truth.view(-1)
        #ground_truth = ground_truth.squeeze()

        mean_gt = torch.mean (ground_truth)
        mean_pred = torch.mean (prediction)
        var_gt = torch.var (ground_truth)
        var_pred = torch.var (prediction)

        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = torch.sum (v_pred * v_gt) / (torch.sqrt(torch.sum(torch.pow(v_pred,2))) * torch.sqrt(torch.sum(torch.pow(v_gt,2))) + 1e-8)

        sd_gt = torch.std(ground_truth)
        sd_pred = torch.std(prediction)
        numerator=2*cor*sd_gt*sd_pred
        denominator=torch.pow(sd_pred,2) + torch.pow(sd_gt,2) + torch.pow(mean_pred - mean_gt,2)
        ccc = numerator/(denominator+ 1e-8)

        return 1-ccc