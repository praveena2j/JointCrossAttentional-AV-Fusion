import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import sys

class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.e = e
        self.reduction = reduction
        self.gamma = 0

    def gaussian(self, mu):
        x = np.linspace(0, 5, 6)
        sig = 0.25
        gauss_func = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        return torch.from_numpy(gauss_func).float().cuda()

    def softlabels(self, labels, classes, value):

        softtarget = torch.zeros(classes, classes)
        softtarget[0, :] = torch.FloatTensor([0.5, 0.3, 0.1, 0.05, 0.025, 0.025])
        softtarget[1, :] = torch.FloatTensor([0.2, 0.5, 0.2, 0.0375, 0.0375, 0.025])
        softtarget[2, :] = torch.FloatTensor([0.0375, 0.2, 0.5, 0.2, 0.0375, 0.025])
        softtarget[3, :] = torch.FloatTensor([0.025, 0.0375, 0.2, 0.5, 0.2, 0.0375])
        softtarget[4, :] = torch.FloatTensor([0.025, 0.0375, 0.0375, 0.2, 0.5, 0.2])
        softtarget[5, :] = torch.FloatTensor([0.025, 0.025, 0.05, 0.1, 0.3, 0.5])

        softlabel = torch.zeros(labels.size(0), classes)
        for i in range(labels.size(0)):
            softlabel[i,:] = softtarget[labels[i], :]
        return softlabel.to(labels.device)

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        #labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)

        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)

        return one_hot



    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format

        """

        #batch_size = target.size()[0]
        #GaussLabels = torch.zeros(target.size(0), length)

        softgausslabels = self.softlabels(target, length, value = 0.5)

        #for sequence in range(target.size(0)):
            #mean = target[sequence]
            #gaussianlabel = self.softlabels(mean.item(), length)
            #gaussianlabel = self.gaussian(mean.item())
            #GaussLabels[sequence, :] = gaussianlabel
        #softgausslabels = self.softmax(GaussLabels)

        #one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        #one_hot += smooth_factor / length

        return softgausslabels.to(target.device)
        #return one_hot.to(target.device)

    def classweights(self, target):
        numDataPoints = target.size(0)

        #print(target)

        #print(numDataPoints)
        weights = np.zeros(6, dtype=np.int)
        class_sample_values, class_sample_count = np.unique(target.cpu().numpy(), return_counts=True)

        #print(class_sample_values)
        #print(class_sample_count)

        weights[class_sample_values] = class_sample_count
        #print(weights)
        #weight = numDataPoints / weights
        weight_new = numDataPoints / (weights + 0.000001)

        #print(weight_new)

        samples_weight = torch.from_numpy(weight_new[target.cpu().numpy()])
        #print(samples_weight)

        return samples_weight

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                    .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                    .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                    .format(x.size()))

        sample_weights = self.classweights(target)
        sample_weights = sample_weights.squeeze()

        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        #target = target.view(-1,1)
        #x = self.log_softmax(x)
        #print(x)
        x = torch.log(x)

        #loss = (x - smoothed_target)**2

        #loss = torch.sum(loss, dim=1)

        #print(loss)

        #loss = loss.view(loss.shape[0], -1)
        #print(target)
        #print(loss)
        #sys.exit()
        #logpt = x.gather(1,target)

        #logpt = logpt.view(-1)
        #pt = Variable(logpt.data.exp())


        #logpt = logpt * Variable(sample_weights.type(x.type()))
        #loss = -1 * (1-pt)**self.gamma * logpt
        #lsmoothing

        loss = torch.sum(- x * smoothed_target, dim=1)
        #loss = -x[range(smoothed_target.shape[0]), smoothed_target].log()
        #print(loss.shape)

        #print(sample_weights.squeeze().shape)
        #print(loss.shape)
        #print(loss)
        loss = loss * sample_weights.type(loss.type())
        #print(loss)
        #sys.exit()

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)

        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')
