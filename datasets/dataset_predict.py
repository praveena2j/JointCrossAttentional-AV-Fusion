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

def default_seq_reader(videoslist, length, stride):
	sequences = []
	sequence_indices = []
	for videos in videoslist:
		video_length = len(videos)
		images = []
		img_labels = []
		arr = []
		for img in videos:
			imgPath, label = img.strip().split(' ')
			img_num = int(os.path.splitext(os.path.split(imgPath)[1])[0][5:])
			img_labels.append(float(label))
			images.append(imgPath)
			arr.append(img_num)
		medfiltered_labels = signal.medfilt(img_labels, 3)
		vid = list(zip(images, medfiltered_labels))
		start = 0
		seq_start = 0
		end = start + 63
		#seq = []
		count = 0
		total = 0
		check_value = 63
		indices = []
		while check_value < 7450:
			sub_arr = arr[:end]
			i = bisect.bisect_right(sub_arr, check_value)
			if (sub_arr[-1] > check_value):
				seq_end = i -1
			else:
				seq_end = i

			if (seq_end > seq_start):
				sequence_indices.append(total)
				sequences.append(vid[seq_start:seq_end])
				count = count + 1
			seq_start = seq_end + 1
			start = sub_arr[i-1] + 1
			end = start + 63
			check_value = check_value + 64
			total = total + 1
	np.save("videoindices.npy", sequence_indices)
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
	def __init__(self, root, fileList, length, flag, stride, list_reader=default_list_reader, seq_reader = default_seq_reader):
		self.root = root
		print(stride)
		self.videoslist = list_reader(fileList)
		self.length = length
		self.stride = stride
		self.sequence_list = seq_reader(self.videoslist, self.length, self.stride)        
		#self.stride = stride
		#self.transform = transform
		#self.dataset = dataset
		#self.loader = loader
		self.flag = flag


	def __getitem__(self, index):
		#for video in self.videoslist:
		seq_path = self.sequence_list[index]
		#img = self.loader(os.path.join(self.root, imgPath), self.flag)
		#if (self.flag == 'train'):
		seq, label = self.load_data_label(seq_path, self.flag)
		label_index = torch.DoubleTensor([label])
		#else:
		#   seq, label = self.load_test_data_label(seq_path)
		#   label_index = torch.LongTensor([label])
		#if self.transform is not None:
		#    img = self.transform(img)
		return seq, label_index

	def __len__(self):
		return len(self.sequence_list)

	def load_data_label(self, SeqPath, flag):
		#print("Loadung training data")
		if (flag == 'train'):
			data_transforms = transforms.Compose([
					transforms.RandomResizedCrop(224),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
			])
		else:
			data_transforms=transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
			]) 
		output = []
		inputs = []
		lab = []
		for image in SeqPath:
			imgPath = image[0]
			label = image[1]
			#imgPath = image.split(" ")[0]
			#label = image.split(" ")[1]
			img = Image.open(imgPath)
			img = img.resize((256,256), Image.ANTIALIAS)
			inputs.append(data_transforms(img).unsqueeze(0))
			lab.append(float(label))
			#print(label)
			#label_idx = float(label)
		label_idx = np.max(lab)
		#print("mean")
		# print(label_idx)
		output_subset = torch.cat(inputs).unsqueeze(0)
		#output.append(output_subset)
		#print(output_subset.size())
		#print(len(output))
		#print(label_idx)
		return output_subset, label_idx
