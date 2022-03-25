import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision import transforms
import torch
from scipy import signal
import bisect
import cv2
import pandas as pd
import utils.videotransforms as videotransforms
import re
import csv

#def default_seq_reader(videoslist, length, stride):
#		sequences = []
		#maxVal = 0.711746
		#minVal = 0.00 #-0.218993
#		for videos in videoslist:
#				video_length = len(videos)
#				if (video_length < length):
#						continue
#				images = []
#				img_labels = []
#				for img in videos:
#						imgPath, label = img.strip().split(' ')
#						img_labels.append(abs(float(label)))
#						#img_labels.append(float(label))
#						images.append(imgPath)
#				medfiltered_labels = signal.medfilt(img_labels, 3)
#				#normalized_labels = (medfiltered_labels-minVal)/(maxVal-minVal)
#				#normalized_labels = (medfiltered_labels-minVal)/(maxVal-minVal)*5
#				vid = list(zip(images, medfiltered_labels))
#				for i in range(0, video_length-length, stride):
#						seq = vid[i : i + length]
#						if (len(seq) == length):
#								sequences.append(seq)
#		#print(len(sequences))
#		return sequences

def default_seq_reader(videoslist, win_length, stride):
	shift_length = stride #length-1
	sequences = []
	csv_data_list = os.listdir(videoslist)
	for video in csv_data_list:
		vid_data = pd.read_csv(os.path.join(videoslist,video))
		video_data = vid_data.to_dict("list")
		images = video_data['img']
		labels_V = video_data['V']
		frame_ids = video_data['frame_id']
		label_array = np.asarray(labels_V, dtype=np.float32)
		medfiltered_labels = signal.medfilt(label_array)
		vid = list(zip(images, medfiltered_labels, frame_ids))
		frameid_array = np.asarray(frame_ids, dtype=np.int32)
		length = len(images)
		start = 0
		end = start + win_length

		if end < length:
			while end < length:
				indices = np.where((frameid_array>=start+1) & (frameid_array<=end))[0]
				frame_id = frameid_array[indices]
				norm_frame_id = frame_id - start -1
				if (len(indices) > 0):
					seq = vid[indices[0]:indices[len(indices)-1]+1]
					#indices = frameid_array[(frameid_array>=start+1) & (frameid_array<=end)]
					#index_start = np.searchsorted(frame_ids, start, 'left')
					#index_end = np.searchsorted(frame_ids, end, 'right')
					#indices = np.arange(index_start, index_end)
					sequences.append([seq, norm_frame_id])
				#else:
				#	sequences.append([])
				start = start + shift_length
				end = start + win_length

			# The last window ends with the trial length, which ensures that no data are wasted.
			start = length - win_length
			end = length
			indices = np.where((frameid_array>=start+1) & (frameid_array<=end))[0]
			if (len(indices) > 0):
				frame_id = frameid_array[indices]
				norm_frame_id = frame_id - start -1
				seq = vid[indices[0]:indices[len(indices)-1]+1]
				sequences.append([seq, norm_frame_id])
			#else:
			#	sequences.append([])
		else:
			end = length
			indices = np.arange(start, end)
			seq = vid[indices[0]:indices[len(indices)-1]+1]
			frame_id = frameid_array[indices]-1
			sequences.append([seq, frame_id])
	return sequences

def default_list_reader(fileList):
	with open(fileList, 'r') as file:
		#print(fileList)
		video_length = 0
		videos = []
		lines = list(file)
		#print(len(lines))
		for i in range(9):
			line = lines[video_length]
			#print(line)
			#line = file.readlines()[video_length + i]
			imgPath, label = line.strip().split(' ')
			find_str = os.path.dirname(imgPath)
			#print(find_str)
			new_video_length = 0
			for line in lines:
				if find_str in line:
					new_video_length = new_video_length + 1
			#print(new_video_length)
			videos.append(lines[video_length:video_length + new_video_length])
			video_length = video_length + new_video_length
			#print(video_length)
	return videos

class ImageList(data.Dataset):
	def __init__(self, root, fileList, length, flag, stride, list_reader=default_list_reader, seq_reader=default_seq_reader):
		self.root = root
		#self.label_path = label_path
		self.videoslist = fileList #list_reader(fileList)
		self.win_length = length
		self.stride = stride
		self.sequence_list = seq_reader(self.videoslist, self.win_length, self.stride)
		#self.stride = stride
		#self.transform = transform
		#self.dataset = dataset
		#self.loader = loader
		self.flag = flag

	def __getitem__(self, index):
		#for video in self.videoslist:
		seq_path, seq_id = self.sequence_list[index]
		#img = self.loader(os.path.join(self.root, imgPath), self.flag)
		#if (self.flag == 'train'):
		seq, label = self.load_data_label(self.root, seq_path, seq_id, self.flag)
		label_index = torch.DoubleTensor([label])
		#else:
		#   seq, label = self.load_test_data_label(seq_path)
		#   label_index = torch.LongTensor([label])
		#if self.transform is not None:
		#    img = self.transform(img)
		return seq, label_index

	def __len__(self):
		return len(self.sequence_list)

	def load_data_label(self, root, SeqPath, seq_id, flag):
		#print("Loadung training data")
		if (flag == 'train'):
			data_transforms = transforms.Compose([videotransforms.RandomCrop(224),
										   videotransforms.RandomHorizontalFlip(),
					#transforms.RandomResizedCrop(224),
					#transforms.RandomHorizontalFlip(),
					#transforms.ToTensor(),
			])
		else:
			data_transforms=transforms.Compose([videotransforms.CenterCrop(224),
				#transforms.Resize(256),
				#transforms.CenterCrop(224),
				#transforms.ToTensor(),
			])
		output = []
		inputs = []
		lab = []
		frame_ids = []
		seq_length = len(SeqPath)
		for image, ids in zip(SeqPath, seq_id):
			imgPath = image[0]
			label = image[1]
			img = cv2.imread(root + imgPath)
			if (img is None):
				img = np.zeros((112, 112, 3), dtype=np.float32)
			w,h,c = img.shape
			#w,h = img.size
			if w == 0:
				continue
			else:
				img = cv2.resize(img, (224, 224))[:, :, [2, 1, 0]]
				#img = img.resize((256, 256), Image.ANTIALIAS)

			img = (img/255.)*2 - 1

			#img = img.resize((256,256), Image.ANTIALIAS)
			#inputs.append(data_transforms(img).unsqueeze(0))
			inputs.append(img)
			lab.append(float(label))
			frame_ids.append(ids)

			#print(label)
			#label_idx = float(label)

		if (len(inputs) <self.win_length):
			imgs = np.zeros((self.win_length, 224, 224, 3), dtype=np.int16)
			lables = np.zeros((self.win_length), dtype=np.int16)
			imgs[frame_ids] = inputs
			lables[frame_ids] = lab
			imgs=np.asarray(imgs, dtype=np.float32)
			targets = np.asarray(lables, dtype=np.float32)
		else:
			imgs=np.asarray(inputs, dtype=np.float32)
			targets = np.asarray(lab, dtype=np.float32)
		#output_subset = torch.cat(inputs)#.unsqueeze(0)
		#output.append(output_subset)
		#print(output_subset.size())
		#print(len(output))
		#print(label_idx)

		#if(imgs.shape[0] != 0):
		imgs = data_transforms(imgs)
		#	return torch.from_numpy(imgs.transpose([3, 0, 1, 2])), lab
		#return output_subset, lab
		return torch.from_numpy(imgs.transpose([3, 0, 1, 2])), targets
		#return torch.from_numpy(imgs.transpose([0, 3, 1, 2])), targets

		#return torch.from_numpy(imgs), lab
		#return output_subset, lab
		#else:
		#	return [], []
