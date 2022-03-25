from models.pytorch_i3d_new import Unit3D
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from utils.functions import ReverseLayerF
import os
import sys
import torch.nn.functional as F
import utils.exp_utils as exp_utils
import numpy as np
from models.temporal_convolutional_model import TemporalConvNet

class I3D_WSDDA(nn.Module):
	def __init__(self, model):
		super(I3D_WSDDA, self).__init__()
		self.i3d_WSDDA = model
		self.predictions = nn.Sequential(
					Unit3D(in_channels=384+384+128+128, output_channels=512,
							 kernel_shape=[1, 1, 1],
							 padding=0,
							 activation_fn=None,
							 use_batch_norm=False,
							 use_bias=True,
							 name='logits'),
					Unit3D(in_channels=512, output_channels=1,
							 kernel_shape=[1, 1, 1],
							 padding=0,
							 activation_fn=None,
							 use_batch_norm=False,
							 use_bias=True,
							 name='logits')
					)

		self.temporal = TemporalConvNet(
			num_inputs=1024, num_channels=[512, 512, 512, 512], kernel_size=5, attention=0,
			dropout=0.1)
		self.regressor = nn.Sequential(nn.Linear(512, 128),
							nn.Dropout(0.5),
							nn.Linear(128, 1),
       						nn.Tanh())

		self.dropout = nn.Dropout(0.5)

	def forward(self, x):
		#batch_size, timesteps, C, H, W = x.size()
		batch_size, C, timesteps, H, W = x.size() ## Inception
		feature = self.i3d_WSDDA.extract_features(x)
		#t = x.size(2)
		#frame_feature = F.interpolate(feature.squeeze(3).squeeze(3), t, mode='linear')#.squeeze(1)
		#features = feature.view(feature.shape[0]*feature.shape[2], -1)#.squeeze()
		features = feature.squeeze(3).squeeze(3)
		#if(neutralframes == 0):
		#	feature = frame_feature.unsqueeze(3).unsqueeze(3)
		#	features = frame_feature.view(frame_feature.shape[0]*frame_feature.shape[2], -1)#.squeeze()
		#else:
		#	for i in range(frame_feature.shape[0]):
		#		frame_feature[i,:,:] = frame_feature[i,:,:] - torch.from_numpy(np.reshape(neutralframes[subids[i]], (1024, 1))).cuda()
		#	feature = frame_feature.unsqueeze(3).unsqueeze(3)
		#	features = frame_feature.view(frame_feature.shape[0]*frame_feature.shape[2], -1)#.squeeze()

			#frame_feature = frame_feature.view(frame_feature.shape[0]*frame_feature.shape[2], -1)#.squeeze()
		#if (flag == 2):  ###  FrameLevel DA
			#feature = self.i3d_WSDDA.extract_features(x)
			#feature = exp_utils.computepeakframe(feature, batch_size, timesteps, numfeat)
			#features = F.interpolate(feature.squeeze(3).squeeze(3), timesteps, mode='linear')#.squeeze(1)
			#features = features.view(features.shape[0]*features.shape[2], -1)#.squeeze()
		#else:
		#	feature = self.i3d_WSDDA.extract_features(x)
		#	if (neutralframes == 1):
		#		numfeat = feature.size(1)
		#		features = exp_utils.computepeakframe(feature, batch_size, timesteps, numfeat)
		#	else:
		#		features = torch.max(feature, dim=2)[0].squeeze(2).squeeze(2)
		#reverse_feature = ReverseLayerF.apply(features, alpha)
		#x = feat.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()

		temp_features = self.temporal(features).transpose(1, 2).contiguous()
		#features = temp_features.contiguous().view(batch_size * timesteps, -1)

		#x = self.regressor(features)
		#output = x.view(batch_size, timesteps, -1)

		#new_feature = self.dropout(feature)
		#class_output = self.predictions(new_feature)
		#else:
		#	class_output = self.target_predictions(new_feature)
		#class_output = self.class_predictions(new_feature)
		#domain_output = self.domain_predictions(reverse_feature)
		return temp_features#, output
