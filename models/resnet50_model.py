from models.temporal_convolutional_model import TemporalConvNet
import math
import os
import torch
from torch import nn
import sys
from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module
from models.resnet_50 import Resnet50_face_sfew_dag

class resnet_TCN(nn.Module):
	def __init__(self, resnet, feat_dim, output_dim, channels=None, attention=0,
				 kernel_size=5, dropout=0.1):
		super().__init__()
		self.spatial = resnet
		self.feat_dim = feat_dim
		self.output_dim = output_dim

		self.channels = channels
		self.kernel_size = kernel_size
		self.attention = attention
		self.dropout = dropout

		self.temporal = TemporalConvNet(
			num_inputs=self.feat_dim, num_channels=self.channels, kernel_size=self.kernel_size, attention=self.attention,
			dropout=self.dropout)
		self.regressor = nn.Sequential(nn.Linear(1024, 256),
							nn.Dropout(0.5),
							nn.Linear(256, self.output_dim))



		#self.regressor = nn.Linear(224, self.output_dim)
		# self.regressor = Sequential(
		#     BatchNorm1d(self.embedding_dim // 4),
		#     Dropout(0.4),
		#     Linear(self.embedding_dim // 4, self.output_dim))

	def forward(self, x):
		num_batches, length, channel, width, height = x.shape
		x = x.view(-1, channel, width, height)
		feat, _ = self.spatial(x)
		_, feature_dim = feat.shape
		x = feat.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()
		x = self.temporal(x).transpose(1, 2).contiguous()
		x = x.contiguous().view(num_batches * length, -1)
		#x = x.view(num_batches, length, feature_dim).contiguous()
		#x = x.contiguous().view(num_batches * length, -1)
		x = self.regressor(x)
		x = x.view(num_batches, length, -1)
		return x
