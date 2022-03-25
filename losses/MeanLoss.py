import torch
import numpy as np
from torch import autograd
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys

class MeanLoss(torch.nn.Module):

    def __init__(self):
        super(MeanLoss,self).__init__()
        self.mseloss = nn.MSELoss().cuda()

    def forward(self, logits, target):
        soft_max = nn.Softmax(dim=1)
        softmax_probabilities = soft_max(logits)

        #_, indices = torch.max(logits, 1)
        #loss_mse = self.mseloss(Variable(indices.type(torch.FloatTensor)).cuda(), target.type(torch.FloatTensor).cuda())

        #logpt = x.gather(1,target)

        #logpt = logpt.view(-1)
        #pt = Variable(logpt.data.exp())

        n = logits.size()
        index = np.arange(1,n[1]+1)
        index_tensor = torch.from_numpy(index).type(torch.FloatTensor).cuda()#

        outputs = (torch.sum(index_tensor * softmax_probabilities, 1)) -1


        target = target.type(torch.FloatTensor).cuda()

        totalloss = torch.sum((torch.round(outputs) - target) ** 2) /(n[0])
        #totalloss = torch.sum((indices - target) ** 2) /(indices.size()[0])

        return totalloss
