
import torch
import torch.nn as nn
import sys
import numpy as np 
class CNN_RNN(nn.Module):
	def __init__(self, model):
		super(CNN_RNN, self).__init__()
		self.cnn = model
		#self.rnn = nn.LSTM(
		#    input_size=4096,
		#    hidden_size=128,
		#    num_layers=2,
		#    batch_first=True,
		#    dropout=0.8)
		
		#self.attention = nn.Sequential(
		#    nn.Linear(self.L, self.D),
		#    nn.Tanh(),
		#    nn.Linear(self.D, self.K)
		#)

		self.gru = nn.GRU(input_size=4096,hidden_size=256, num_layers=2, batch_first=True)
		self.classify = nn.Linear(256, 1)
		self.dropout = nn.Dropout(p=0.7)

	def forward(self, x):
		#print(x.size())
		batch_size, timesteps, C, H, W = x.size()
		
		#print("batch size")
		#print(batch_size)

		#print("time steps")
		#print(timesteps)
		c_in = x.view(batch_size*timesteps, C, H, W)
		# x = x.view(-1, self.num_flat_features(x))
		c_out = self.cnn(c_in)
		#mean_face = torch.mean(c_out, 0)
		#mean_face = np.mean(c_out.detach().cpu().numpy(), axis=0)
		
		#c_out_mean = c_out - mean_face[None, :, :, :]
		
		#for i in range(c_out.shape[0]):
		#	frame_feature[i,:,:] = frame_feature[i,:,:] - torch.from_numpy(np.reshape(neutralframes[subids[i]], (1024, 1))).cuda()

		#A = self.attention(c_out) 
		#A = torch.transpose(A, 1, 0)  # KxN
		#A = F.softmax(A, dim=1)  # softmax over N

		#print(self.num_flat_features(c_out))
		

		r_in = c_out.view(batch_size, -1, self.num_flat_features(c_out))

		self.gru.flatten_parameters()
		#r_out, (h_n, h_c) = self.lstm(r_in)
		
		r_out, h_n = self.gru(r_in)
		r_out = self.dropout(r_out)

		#print(r_out.shape)
		rnn_out = r_out.contiguous().view(batch_size*timesteps, r_out.shape[2])
	 
		#print(r_out)
		#print(r_out[:, -1, :].shape)
		#rnn_out = torch.mean(r_out,1)

		#rnn_out, _ = torch.max(r_out,1)
		#print(rnn_out.shape)
		#print(rnn_out)
		#sys.exit()
		#print(r_out)
		#print(r_out[:,-1,:])
		#sys.exit()
		#rnn_out = r_out[:, -1, :]
		#print(rnn_out.shape)
		#sys.exit()
		# print(trch_mean.shape)
		r_out2 = self.classify(rnn_out)

		# x = x.view(-1, self.num_flat_features(x))
		# x = F.relu(self.fc1(x))

		# x = F.dropout(x, training=self.training)
		# return F.log_softmax(x, dim=1)
		return r_out2

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
