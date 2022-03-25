from __future__ import print_function
import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
#import torch.utils.data
import torch.nn.functional as F
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#from torchsummary import summary
import torchvision.models as models
# from models import *
from collections import OrderedDict
from torch.autograd import Variable
# import scipy as sp
from scipy import signal
from models.resnet_50 import Resnet50_face_sfew_dag
from models.resnet50_model import resnet_TCN
from models.pytorch_i3d_new import InceptionI3d
from models.I3DWSDDA import I3D_WSDDA
from models.CNN_LSTM import CNN_RNN
from models.Vgg_vd_face_fer_dag import Vgg_vd_face_fer_dag
from train import train
from val import validate
from test import Test
import logging
#import models.resnet as ResNet
import utils
import matplotlib.pyplot as plt
import numpy as np
# import cv2
from models.cam import CAM
from models.tsav import TwoStreamAuralVisualModel
import sys
#from fer import FER2013
#from load_imglist import ImageList
from datasets.dataset_new import ImageList
import math
from losses.CCC import CCC
from losses.CCCLoss import CCCLoss
import wandb
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#wandb.init(settings=wandb.Settings(start_method="fork"), project='Audio Visual Fusion')
from models.vggish_pytorch.vggish import VGGish

parser = argparse.ArgumentParser(description='PyTorch Deep WSDAOR')
parser.add_argument('--arch', '-a', metavar='ARCH', default='WSDA-OR')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
					metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
					help='root path of face images (default: none).')
parser.add_argument('-seq_l','--seq-length', default=64, type=int, metavar='N',
					help='sequence length for lstm')
parser.add_argument('-stride','--stride-length', default=64, type=int, metavar='N',
					help='stride length for lstm')
parser.add_argument('--train_list', default='', type=str, metavar='PATH',
					help='path to training list (default: none)')
parser.add_argument('--val_list', default='', type=str, metavar='PATH',
					help='path to validation list (default: none)')
parser.add_argument('--wavs_list', default='', type=str, metavar='PATH',
					help='path to wav files (default: none)')
parser.add_argument('--time_list', default='', type=str, metavar='PATH',
					help='path to timestamps (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
					help='save root path for features of face images.')
parser.add_argument('--num_classes', default=79077, type=int,
					metavar='N', help='number of classes (default: 79077)')
args = parser.parse_args()

best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_Val_acc = 0  # best PrivateTest accuracy
best_Val_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 5  # 50
learning_rate_decay_every = 2 # 5
learning_rate_decay_rate = 0.8 # 0.9
total_epoch = 30

TrainingAccuracy = []
ValidationAccuracy = []
#def init_weights(m):
#    if type(m) == nn.Linear:
#        torch.nn.init.xavier_uniform(m.weight)
#        m.bias.data.fill_(0.01)

ts = time.time()
Logfile_name = "LogFiles/" + "log_file.log"
logging.basicConfig(filename=Logfile_name, level=logging.INFO)

SEED = 0
### Using seed for deterministic perfromVisual_model_withI3Dg order
if (SEED == 0):
	torch.backends.cudnn.benchmark = True
else:
	print("Using SEED")
	random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(SEED)

class PadSequence:
	def __call__(self, batch):
		sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
		sequences = [x[0] for x in sorted_batch]
		aud_sequences = [x[1] for x in sorted_batch]
		for aud in aud_sequences:
			print(aud.shape)
		sys.exit()
		#sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
		aud_sequences_padded = torch.nn.utils.rnn.pad_sequence(aud_sequences, batch_first=True)
		labels = [x[2] for x in sorted_batch]
		#vis_seq_padded = sequences_padded.permute(0,4,1,2,3)
		audio_sequences = torch.stack(aud_sequences_padded)
		return sequences, audio_sequences, labels

if not os.path.isdir("SavedWeights"):
	os.makedirs("SavedWeights")

path = "SavedWeights"
#resnet = Resnet50_face_sfew_dag()
#resnet.load_state_dict(torch.load('PretrainedWeights/resnet50_face_sfew_dag.pth'))
#cnn_lstm_model = resnet_TCN(resnet, feat_dim=2048, output_dim=1, channels=[1024, 1024, 1024, 1024], attention=0,
#				 kernel_size=5, dropout=0.1)
#cnn_lstm_model.cuda()
#cnn_lstm_model = nn.DataParallel(cnn_lstm_model)
#cudnn.benchmark = True

i3d = InceptionI3d(400, in_channels=3)
#i3d.load_state_dict(torch.load('PretrainedWeights/rgb_imagenet.pt'))
cnn_lstm_model = I3D_WSDDA(i3d)
cnn_lstm_model.cuda()
cnn_lstm_model = nn.DataParallel(cnn_lstm_model)
cnn_lstm_model.load_state_dict(torch.load('PretrainedWeights/Val_model_valence_cnn_lstm_mil_64_new.t7')['net'])

visualmodel_acc = torch.load('PretrainedWeights/Val_model_valence_cnn_lstm_mil_64_new.t7')['best_Val_acc']
print(visualmodel_acc)
for param in cnn_lstm_model.module.i3d_WSDDA.parameters():  # children():
	param.requires_grad = False

model_path = '../ABAW2020TNT/aff2model_tntsub4/model2/TSAV_Sub4_544k.pth.tar' # path to the model
model = TwoStreamAuralVisualModel(num_channels=4)
saved_model = torch.load(model_path)
model.load_state_dict(saved_model['state_dict'])
model = model.to('cuda')
for p in model.children():
    audio_model = p
    break
#audio_model = nn.DataParallel(audio_model)
#for param in audio_model.module.parameters():  # children():
for param in audio_model.parameters():  # children():
	param.requires_grad = False

#model_urls = {
#'vggish': 'https://github.com/harritaylor/torchvggish/'
#		  'releases/download/v0.1/vggish-10086976.pth',
#'pca': 'https://github.com/harritaylor/torchvggish/'
#	   'releases/download/v0.1/vggish_pca_params-970ea276.pth'
#}
#audio_model = VGGish(model_urls)
#audio_model = nn.DataParallel(audio_model)
#for param in audio_model.module.parameters():  # children():
#	param.requires_grad = False

print('==> Preparing data..')
label_file = '../../SpeechEmotionRec/ratings_gold_standard/ratings_gold_standard/valence/'

traindataset = ImageList(root=args.root_path, fileList=args.train_list, audList=args.wavs_list,
              length=256, flag='train', stride=1, dilation = 4, subseq_length = 32)
trainloader = torch.utils.data.DataLoader(
    			traindataset,
  		batch_size=96, shuffle=True, #collate_fn=PadSequence(),
		num_workers=4, pin_memory=True, drop_last = True)

valdataset = ImageList(root=args.root_path, fileList=args.val_list, audList=args.wavs_list,
              length=256, flag='val', stride=1, dilation = 4, subseq_length = 32)
valloader = torch.utils.data.DataLoader(
				valdataset,
		batch_size=96, shuffle=False, #collate_fn=PadSequence(),
		num_workers=4, pin_memory=True, drop_last = True)

#testdataset = ImageList(root=args.root_path, fileList=args.val_list, audList=args.wavs_list,
#              length=256, flag='val', stride=4, dilation = 4, subseq_length = 32)
#testloader = torch.utils.data.DataLoader(
#				valdataset,
#		batch_size=96, shuffle=False, #collate_fn=PadSequence(),
#		num_workers=4, pin_memory=True, drop_last = True)

print("Number of Train samples:" + str(len(traindataset)))
print("Number of Val samples:" + str(len(valdataset)))

cam = CAM().cuda()
#cam.load_state_dict(torch.load('SavedWeights/Val_model_valence_cnn_lstm_mil_64_new.t7')['net'])
#best_Val_acc = torch.load('SavedWeights/Val_model_valence_cnn_lstm_mil_64_new.t7')['best_Val_acc']
#cudnn.benchmark = True
criterion = CCC().cuda()
optimizer = torch.optim.Adam(cam.parameters(),# filter(lambda p: p.requires_grad, multimedia_model.parameters()),
								args.lr)
#optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, cnn_lstm_model.parameters()),
#								args.lr,
#								momentum=args.momentum,
#								weight_decay=args.weight_decay)

#optimizer = torch.optim.Adam(model.parameters(), lr= 0.001  , amsgrad=True)
cnt = 0
for epoch in range(start_epoch, total_epoch):
	#adjust_learning_rate(optimizer, epoch)
	#adjust_learning_rate(optimizer, epoch)

	logging.info("Epoch")
	logging.info(epoch)
	#if cnt == 0:
	# train for one epoch
	Training_loss, Training_acc = train(trainloader, cnn_lstm_model, audio_model, criterion, optimizer, epoch, cam)
	#cnt = cnt + 1
	# evaluate on validation set
	#Training_acc = 0.0
	Valid_loss, Valid_acc = validate(valloader, cnn_lstm_model, audio_model, criterion, epoch, cam)
	#Test(PrivateTestloader , original_model, criterion, epoch)
	TrainingAccuracy.append(Training_acc)
	ValidationAccuracy.append(Valid_acc)

	logging.info('TrainingAccuracy:')
	logging.info(TrainingAccuracy)

	logging.info('ValidationAccuracy:')
	logging.info(ValidationAccuracy)

	if Valid_acc > best_Val_acc:
		print('Saving..')
		print("best_Val_acc: %0.3f" % Valid_acc)
		state = {
			'net': cam.state_dict() ,
			'best_Val_acc': Valid_acc,
			'best_Val_acc_epoch': epoch,
		}
		if not os.path.isdir(path):
			os.mkdir(path)
		torch.save(state, os.path.join(path,'Val_model_valence_cnn_lstm_mil_64_new.t7'))
		best_Val_acc = Valid_acc
		best_Val_acc_epoch = epoch

#print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
#print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_Val_acc)
print("best_PrivateTest_acc_epoch: %d" % best_Val_acc_epoch)
