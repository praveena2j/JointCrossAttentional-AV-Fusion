from __future__ import print_function
import argparse
import os
import shutil
import time

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
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from utils.utils import clip_gradient

import utils.utils as utils
from utils.exp_utils import pearson
from EvaluationMetrics.ICC import compute_icc
from EvaluationMetrics.cccmetric import ccc

from utils.utils import Normalize
from utils.utils import calc_scores
import logging
# import models.resnet as ResNet
#import utils
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import sys
import math
from losses.CCC import CCC
import wandb
learning_rate_decay_start = 5  # 50
learning_rate_decay_every = 2 # 5
learning_rate_decay_rate = 0.8 # 0.9
total_epoch = 30
lr = 0.001
scaler = torch.cuda.amp.GradScaler()

def train(train_loader, visual_model, audio_model, criterion, optimizer, epoch, cam):
	print('\nEpoch: %d' % epoch)
	global Train_acc
	#wandb.watch(audiovisual_model, log_freq=100)
	#wandb.watch(cam, log_freq=100)

	# switch to train mode
	#audiovisual_model.train()
	audio_model.eval()
	visual_model.eval()
	cam.train()

	train_loss = 0
	correct = 0
	total = 0
	running_loss = 0
	running_accuracy = 0
	out = []
	tar = []

	if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
		frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
		decay_factor = learning_rate_decay_rate ** frac
		current_lr = lr * decay_factor
		utils.set_lr(optimizer, current_lr)  # set the decayed rate
	else:
		current_lr = lr
	print('learning_rate: %s' % str(current_lr))
	logging.info("Learning rate")
	logging.info(current_lr)
	#torch.cuda.synchronize()
	#t1 = time.time()
	for batch_idx, (visualdata, audiodata, labels) in tqdm(enumerate(train_loader),
														 total=len(train_loader), position=0, leave=True):

		#if(batch_idx > 2):#int(65844/64)):
		#	break
		#torch.cuda.synchronize()
		#t2 = time.time()
		#print('data loading time', t2-t1)
		optimizer.zero_grad(set_to_none=True)
		audiodata = audiodata.cuda()
		visualdata = visualdata.cuda()#permute(0,4,1,2,3).cuda()
		labels = labels.cuda()
		#visuallabel = visuallabel.squeeze(1).type(torch.FloatTensor).cuda()
		#print("training started")
		#torch.cuda.synchronize()
		#t3 = time.time()

		with torch.cuda.amp.autocast():
			with torch.no_grad():
				b, c, seq_t, subseq_t, h, w = visualdata.size()
				#visualdata = visual_data.view(b, c, -1, sub_seq_len, h, w)
				visual_feats = []
				aud_feats = []

				#vis_data = visualdata.view(b*visualdata.shape[2], c, subseq_t ,h , w)
				#visualfeatures, _ = visual_model(vis_data)
				#visual_feat = visualfeatures.view(b, -1, visualfeatures.shape[1])
				#print(visfin_test.shape)
				#print(visual_feat.shape)
				for i in range(visualdata.shape[0]):
					vis_dat = visualdata[i, :, :, :,:,:].transpose(0,1)
					visualfeat = visual_model(vis_dat)
					visualfeat, _ = torch.max(visualfeat,1)
					#visual_feat = visualfeat.view(b, -1, visualfeat.shape[1])
					visual_feats.append(visualfeat)
					#aud_data = audiodata.view(audiodata.shape[0]*audiodata.shape[1], audiodata.shape[2], audiodata.shape[3]).unsqueeze(1)
					aud_data = audiodata[i,:,:,:]#.unsqueeze(1)
					aud_feat = audio_model(aud_data)
					#print(aud_feat.shape)
					#audio_feat = aud_feat.view(b, -1, aud_feat.shape[1])
					aud_feats.append(aud_feat) #.squeeze(3))
				#print(audio_feat.shape)
				visual_feat = torch.stack(visual_feats)#.squeeze(3).squeeze(3).squeeze(3)#.transpose(1,2)
				#visual_feat = visual_feat.view(visual_feat.shape[0]*visual_feat.shape[1], -1)
				#print(visual_feat.shape)
				#torch.cuda.synchronize()
				#t4 = time.time()
				#print('visual feature extraction time', t4-t3)

				#torch.cuda.synchronize()
				#t5 = time.time()
				#aud_feats = []
				#print(audiodata.shape)
				#for i in range(audiodata.shape[0]):
				#	aud_data = audiodata[i,:,:,:].unsqueeze(1)
				#	audio_feat, audio_out = audio_model(aud_data)
				#	aud_feats.append(audio_feat.squeeze(3))
				audio_feat = torch.stack(aud_feats)#.squeeze(3)#.transpose(1,2)
				#print(audio_feat.shape)
				#print(visual_feat.shape)
				#torch.cuda.synchronize()
				#t6 = time.time()
				#print('audio feature extraction time', t6-t5)
				#audio_feat = audio_feat.squeeze(3)#.transpose(1,2)

				#visual_feat = torch.max(visual_feat, dim = 2)[0]#.squeeze(2).squeeze(2)
				#audio_feat = torch.max(audio_feat, dim = 2)[0]#.squeeze(2).squeeze(2)

				#print("features extracted")
				#audio_feat_norm = F.normalize(audio_feat, p=2, dim=2, eps=1e-12)
				#visual_feat_norm = F.normalize(visual_feat, p=2, dim=2, eps=1e-12)

			audiovisual_outs = cam(audio_feat, visual_feat)
			#audio_attfeat, visual_attfeat = cam(audio_feat_norm, visual_feat_norm)
			#audiovisual_outs = audiovisual_model(audio_attfeat, visual_attfeat)

			outputs = audiovisual_outs.view(-1, audiovisual_outs.shape[0]*audiovisual_outs.shape[1])
			targets = labels.view(-1, labels.shape[0]*labels.shape[1])#.cuda()

			loss = criterion(outputs, targets)
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		out = np.concatenate([out, outputs.squeeze(0).detach().cpu().numpy()])
		tar = np.concatenate([tar, targets.squeeze(0).detach().cpu().numpy()])

		if torch.isnan(loss):
			print(outputs)
			print(targets)
			print(loss)
			sys.exit()

		#if batch_idx % 100 == 0:
		#	wandb.log({"train_loss": loss})
		#	pass

	#pred, tar = Normalize(out, tar)

	if (len(tar) > 1):
		train_acc = ccc(out, tar)
	else:
		train_acc = 0
	print("Train Accuracy")
	#wandb.log({"train_acc": train_acc})
	print(train_acc)
	#xcorr_weights = 0
	return (loss), (train_acc)
