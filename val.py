from __future__ import print_function
import argparse
import os
import shutil
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
#import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from torchsummary import summary
import torchvision.models as models
# from models import *
from collections import OrderedDict
from torch.autograd import Variable
# import scipy as sp
from scipy import signal

from utils.utils import Normalize
from utils.utils import calc_scores
import logging
# import models.resnet as ResNet
import utils
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import sys
from EvaluationMetrics.cccmetric import ccc

import math
from losses.CCC import CCC
#import wandb


def validate(val_loader, visual_model, audio_model, criterion, epoch, cam):
	# switch to evaluate mode
	global Val_acc
	global best_Val_acc
	global best_Val_acc_epoch
	#model.eval()
	audio_model.eval()
	visual_model.eval()
	cam.eval()

	PrivateTest_loss = 0
	correct = 0
	total = 0
	running_val_loss = 0
	running_val_accuracy = 0

	out = []
	tar = []
	#torch.cuda.synchronize()
	#t7 = time.time()

	for batch_idx, (visual_data, audiodata, labels) in tqdm(enumerate(val_loader),
														 total=len(val_loader), position=0, leave=True):
		#if(batch_idx > 2):#int(65844/64)):
		#	break

		#torch.cuda.synchronize()
		#t8 = time.time()
		#print('data loading time', t8-t7)

		audiodata = audiodata.cuda()
		visualdata = visual_data.cuda()

		#torch.cuda.synchronize()
		#t9 = time.time()

		with torch.no_grad():
			b, c, seq_t, subseq_t, h, w = visualdata.size()
			#sub_seq_len = 16
			#visualdata = visual_data.view(b, c, -1, sub_seq_len, h, w)
			visual_feats = []
			aud_feats = []
			for i in range(visualdata.shape[0]):
				vis_dat = visualdata[i, :, :, :,:,:].transpose(0,1)
				visualfeat = visual_model(vis_dat)
				visualfeat, _ = torch.max(visualfeat,1)
				visual_feats.append(visualfeat)

				aud_data = audiodata[i,:,:,:]#.unsqueeze(1)
				audio_feat = audio_model(aud_data)
				aud_feats.append(audio_feat) #.squeeze(3))

			visual_feat = torch.stack(visual_feats)#.squeeze(3).squeeze(3).squeeze(3)#.transpose(1,2)
			audio_feat = torch.stack(aud_feats)#.squeeze(3)#.transpose(1,2)
			#torch.cuda.synchronize()
			#t8 = time.time()

			#audio_feat, audio_out = audio_model(audiodata)
			#audio_feat = audio_feat.squeeze(3)

			#audio_feat, audio_out = audio_model(audiodata)
			#visualfeat, visual_out = visual_model(visualdata)#.unsqueeze(0))
			#visual_feat = visualfeat.squeeze(2).squeeze(2).squeeze(2)
			#visual_feat = torch.max(visualfeat, dim = 2)[0].squeeze(2).squeeze(2)

			#vis_data = visualdata.view(b*visualdata.shape[2], c, subseq_t ,h , w)
			#visualfeatures, _ = visual_model(vis_data)
			#visual_feat = visualfeatures.view(b, -1, visualfeatures.shape[1])

			#aud_data = audiodata.view(audiodata.shape[0]*audiodata.shape[1], audiodata.shape[2], audiodata.shape[3]).unsqueeze(1)
			#aud_feat, audio_out = audio_model(aud_data)
			#audio_feat = aud_feat.view(b, -1, aud_feat.shape[1])

			#print(audio_feat.shape)
			#print(visual_feat.shape)

			#audio_feat_norm = F.normalize(audio_feat, p=2, dim=2, eps=1e-12)
			#visual_feat_norm = F.normalize(visual_feat, p=2, dim=2, eps=1e-12)
			#audio_attfeat, visual_attfeat = cam(audio_feat, visual_feat)
			#audiovisual_outs = model(audio_feat_norm, visual_feat_norm)

			audiovisual_outs = cam(audio_feat, visual_feat)
			outputs = audiovisual_outs.view(-1, audiovisual_outs.shape[0]*audiovisual_outs.shape[1])
			targets = labels.view(-1, labels.shape[0]*labels.shape[1]).cuda()

		val_loss = criterion(outputs, targets)

		#if batch_idx % 100 == 0:
		#	#wandb.log({"val_loss": val_loss})

		out = np.concatenate([out, outputs.squeeze(0).detach().cpu().numpy()])
		tar = np.concatenate([tar, targets.squeeze(0).detach().cpu().numpy()])

	#pred, tar = Normalize(out, tar)
	if (len(tar) > 1):
		Val_acc = ccc(out, tar)
	else:
		Val_acc = 0

	print("Val Accuracy")
	#wandb.log({"Val_acc": Val_acc})

	print(Val_acc)
	return val_loss, (Val_acc)
