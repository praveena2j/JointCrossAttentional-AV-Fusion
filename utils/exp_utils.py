#from visdom import Visdom
from numpy import linalg as LA
import numpy as np
import torch
import matplotlib.pyplot as plt
import os.path as osp
import os
import torch.nn.init as init
import torch
import torch.nn as nn
#from visdom import Visdom
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from PIL import Image
import sys
import cv2

def make_weights_for_balanced_classes(images, nclasses):
	count = [0] * nclasses
	trainsequences = []
	trainlabels = []
	for seq in images:
		trainsequence = []
		trainlabel = []
		for frame in seq:
			sequence = frame.strip().split(' ')[0]
			label = frame.strip().split(' ')[1]
			trainsequence.append(sequence)
			trainlabel.append(int(label))
		trainsequences.append([trainsequence, max(trainlabel)])

	for item in trainsequences:
		count[item[1]] += 1
	weight_per_class = [0.] * nclasses
	print(count)
	N = float(sum(count))
	for i in range(nclasses):
		weight_per_class[i] = N/float(count[i])
	print(weight_per_class)
	weight = [0] * len(images)
	for idx, val in enumerate(trainsequences):
		weight[idx] = weight_per_class[val[1]]
	return weight


def default_list_train_val_reader(label_path, fileslist, num_subjects, data_domain):
	videos = []
	video_length = 0
	for filelist in fileslist:
		with open(label_path + filelist, 'r') as file:
			lines = list(file)
			while (video_length < len(lines)):
				line = lines[video_length]
				imgPath = line.strip().split(' ')[0]
				if (data_domain == "source"):
					find_str = os.path.dirname(imgPath)
				else:
					find_str = os.path.dirname(os.path.dirname(imgPath))
				new_video_length = 0
				for line in lines:
					if find_str in line:
						new_video_length = new_video_length + 1
					else:
						break
				videos.append(lines[video_length:video_length + new_video_length])
				lines = lines[video_length + new_video_length :]
	trainvideos = videos[0:num_subjects]
	valvideos = videos[num_subjects:len(videos)]
	return trainvideos, valvideos

def pearson(x, y):
	# Assume len(x) == len(y)
	n = len(x)
	print(n)
	sum_x = float(sum(x))
	sum_y = float(sum(y))
	sum_x_sq = sum(map(lambda x: pow(x, 2), x))
	sum_y_sq = sum(map(lambda x: pow(x, 2), y))
	psum = sum(list(map(lambda x, y: x * y, x, y)))
	num = psum - (sum_x * sum_y / n)
	den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
	if den == 0:
		# print(x)
		# print(y)
		ans = len(list(set(x).intersection(y))) / len(x)
		# print(ans)
		return ans
	return num / den

def computepeakframeinbatch(numpy_tesnsors, timesteps, numfeat):
	mean_face = np.zeros((1, numfeat))
	count = 0

	for image in range(timesteps):
		numpy_image = numpy_tesnsors[image].reshape((1, numfeat))
		mean_face = np.add(mean_face, numpy_image)
		count = count + 1
	# print(count)
	mean_face = np.divide(mean_face, float(timesteps)).flatten()
	normalised_training_tensor = np.ndarray(shape=(timesteps, numfeat))

	for i in range(timesteps):
		numpy_image = numpy_tesnsors[i].reshape((1, numfeat))
		normalised_training_tensor[i] = np.subtract(numpy_image, mean_face)
	# print(normalised_training_tensor.shape)
	distances = LA.norm(normalised_training_tensor, axis=1)
	print(max(distances))

	result = np.where(distances == np.amax(distances))
	print(result)
	return result


def computepeakframe(r_in, batch_size, timesteps, numfeat):
	results = []
	maxdistances = []
	batchdistances = []
	max_features = []
	for batch in range(batch_size):
		im = r_in[batch, :, :]
		numpy_tesnsors = im.data.cpu().numpy()
		mean_face = np.zeros((1, numfeat))
		count = 0
		for image in range(timesteps):
			numpy_image = numpy_tesnsors[:,image].reshape((1, numfeat))
			mean_face = np.add(mean_face, numpy_image)
			count = count + 1

		# print(count)
		mean_face = np.divide(mean_face, float(timesteps))#.flatten()
		normalised_training_tensor = np.ndarray(shape=(timesteps, numfeat))

		mean_distances = []
		for i in range(timesteps):
			numpy_image = numpy_tesnsors[:,i].reshape((1, numfeat))
			normalised_training_tensor[i] = np.subtract(numpy_image, mean_face)
		distances = LA.norm(normalised_training_tensor, axis=1)
		result = np.where(distances == np.amax(distances))
		res = result[0][0]
		max_features.append(im[:,res])
	max_features = torch.stack(max_features)
	return max_features

class detection_collate:
	def __call__(self, batch):
		targets = []
		imgs = []
		for sample in batch:
			inp = torch.squeeze(sample[0])
			print(inp.size())
			imgs.append(inp)
			targets.append(sample[1].float())
			seq = torch.stack(imgs, 0)
			tar = torch.stack(targets, 0)
		return seq, tar



class PadSequence:
	def __call__(self, batch):
		# Let's assume that each element in "batch" is a tuple (data, label).
		# Sort the batch in the descending order
		sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
		# Get each sequence and pad it
		#for r in sorted_batch:
		#    print(r[0].size())
		#    print(r[1].size())
		#sys.exit()
		sequences = [x[0].permute(1,2,3,0) for x in sorted_batch]
		#sequences = [x[0] for x in sorted_batch]
		#print(sequences[0].size())

		#print(type(sequences[0]))

		#print(sequences.size())
		sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences)
		#print(sequences_padded.size())
		sequences_orig = sequences_padded.permute(1,4,0,2,3)
		#print(sequences_orig.size())
		#sys.exit()
		# Also need to store the length of each sequence
		# This is later needed in order to unpad the sequences
		#lengths = torch.LongTensor([len(x) for x in sequences])
		lengths = torch.LongTensor([len(x) for x in sequences_orig])

		# Don't forget to grab the labels of the *sorted* batch
		#labels = torch.LongTensor(list(map(lambda x: x[1], sorted_batch)))
		labels = [torch.max(x[1]) for x in sorted_batch]
		#print(labels)
		#print(labels[0].size())
		#print(type(labels))
		#print(len(labels))

		stacked_tensor = torch.stack(labels)
		#print(stacked_tensor.size())
		#print(sequences_orig.size())

		return sequences_orig, lengths, labels
		#return sequences_padded, lengths, labels


def imscatter(features, images, saveimage, ax=None, zoom=0.3):
	fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
	for feature, image  in zip(features, images):
		im = OffsetImage(image, zoom=zoom)
		ab = AnnotationBbox(im, (feature[0], feature[1]), xycoords='data', frameon=False)
		ax1.add_artist(ab)
	#ax.update_datalim(np.column_stack([x, y]))
	ax1.autoscale()
	ax1.scatter(features[:, 0], features[:, 1])
	plt.savefig(saveimage)

def plot_features_DA(source_frame_features, source_features, target_features, source_labels, num_classes, epoch, path, prefix, subject):

	tsne = TSNE(n_components=2, perplexity=40)

	sourcefeatures = tsne.fit_transform(source_features)
	sourceframe_features = tsne.fit_transform(source_frame_features)
	targetfeatures = tsne.fit_transform(target_features)


	fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)

	"""Plot features on 2D plane.
	Args:
		features: (num_instances, num_features).
		labels: (num_instances).
	"""

	sourcelabels = [int(x+.5) for x in source_labels]
	#sourcelabels = sourcelabels.tolist()
	#sourcelabels = [print(x) for x in source_labels]
	colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
	size = [10, 30, 60, 90, 120, 150]
	#markers = ['.']
	#ax2.autoscale()
	for label_idx in range(num_classes):
		ax2.scatter(
			sourceframe_features[np.array(sourcelabels) == label_idx, 0],
			sourceframe_features[np.array(sourcelabels) == label_idx, 1],
			alpha=0.5,
			c=colors[label_idx],
			s=size[label_idx],
			marker='.',
		)

	labels = np.concatenate([np.zeros(sourcefeatures.shape[0]), np.ones
	(targetfeatures.shape[0])], 0)
	all_labels = [int(x) for x in labels]

	all_features = np.concatenate([sourcefeatures, targetfeatures], 0)
	colors = ['C0', 'C1' ]
	size = [5, 5 ]
	#markers = ['*']
	#ax1.autoscale()
	for label_idx in range(2):
		ax1.scatter(
			all_features[np.array(all_labels) == label_idx, 0],
			all_features[np.array(all_labels) == label_idx, 1],
			alpha=0.5,
			c=colors[label_idx],
			s=size[label_idx],
			marker='o',
		)
	ax2.legend(['0', '1', '2', '3', '4', '5'], loc='best')
	ax2.set_title('Prediction Learning')
	ax1.set_title('Domain Adaptation')
	ax1.legend(['0', '1'], loc='best')
	save_name = osp.join(path,  "Epoch"  + str(epoch) +'.png')
	plt.savefig(save_name, bbox_inches='tight')
	#plt.savefig(save_name)
	plt.close(fig)


def plot_features(images, cnn_features, labels, num_classes, epoch, path, prefix, subject):

	tsne = TSNE(n_components=2, perplexity=40)

	if (prefix == 'peakframe'):
		batch, sequence = epoch

		batch_dir = path + "/" + 'Batch' + str(batch)
		seq_dir = batch_dir + "/" + 'Seq' + str(sequence)

		if not osp.exists(batch_dir):
			os.mkdir(batch_dir)
		if not osp.exists(seq_dir):
			os.mkdir(seq_dir)

		orig_features, peak_index = cnn_features
		#print(orig_features.shape)
		#print(orig_result.shape)
		features = tsne.fit_transform(orig_features)
		result = features[peak_index, :]

		#print(features.shape)
		#print(result.shape)
		#sys.exit()

		fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
		#ax1 = fig.add_subplot(2, 1, 1)
		#ax2 = fig.add_subplot(2, 2, 1, sharey=ax1)
		count = 0
		for feature, image  in zip(features, images):
			im = OffsetImage(image, zoom=0.2)
			ab = AnnotationBbox(im, (feature[0], feature[1]), xycoords='data', frameon=False)
			ax1.add_artist(ab)
			save_image = osp.join(path,  "Batch"  +str(batch) , "Seq" + str(sequence)  , 'Image' + str(count) +'.png')
			image.save(save_image)
			count = count + 1
		#ax.update_datalim(np.column_stack([x, y]))
		ax1.autoscale()
		ax1.scatter(features[:, 0], features[:, 1])

		#result = computepeakframeinbatch(features, features.shape[0], features.shape[1])

		save_name = osp.join(path,  "Batch"  +str(batch) + "Sequence" + str(sequence)  + 'Features_MaxLabel' + str(max(labels)) +'.png')
		#print(labels)
		#print(type(images))
		#print(len(images))
		#print(images[len(images)-1].size)

		#rnn_last_image = images[len(images)-1]
		#save_image = osp.join(path,  "Batch"  + str(epoch) + 'Lastimage_MaxLabel' + str(max(labels)) +'.png')
		#rnn_last_image.save(save_image)
		#sys.exit()
	else:
		fig, ax2 = plt.subplots()
		save_name = osp.join(path,  "Epoch"  + str(epoch) + '_MaxLabel' + str(max(labels)) +'.png')
		features = tsne.fit_transform(cnn_features)
	#plt.savefig(saveimage)

	"""Plot features on 2D plane.
	Args:
		features: (num_instances, num_features).
		labels: (num_instances).
	"""
	#print(labels)

	# print("res")
	#print(max(labels))
	# print(labels[result])
	# colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
	colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
	size = [10, 30, 60, 90, 120, 150]
	markers = ['.']
	for label_idx in range(num_classes):
		ax2.scatter(
			features[labels == label_idx, 0],
			features[labels == label_idx, 1],
			alpha=0.5,
			c=colors[label_idx],
			s=size[label_idx],
			marker='.',
		)

	if (prefix == 'peakframe'):

		rnn_last = features[features.shape[0]-1, :]
		#result = computepeakframeinbatch(features, features.shape[0], features.shape[1])
		max_feature = np.amax(features, axis=0)
		avg_feature = np.mean(features, axis=0)


		A = result[0], rnn_last[0]#, max_feature[0], avg_feature[0]
		B = result[1], rnn_last[1]#, max_feature[1], avg_feature[1]
		annotations = ['peak', 'rnn', 'max', 'avg']

		x0,x1=ax2.get_xlim()
		y0,y1=ax2.get_ylim()

		for ii, ind in enumerate(np.argsort(B)):
			x = A[ind]
			y = B[ind]
			xPos = x1 + .02 * (x1 - x0)
			yPos = y0 + ii * (y1 - y0)/(len(B) - 1)
			ax2.annotate(annotations[ind],#label,
			  xy=(x, y), xycoords='data',
			  xytext=(xPos, yPos), textcoords='data',
			  arrowprops=dict(
							  connectionstyle="arc3,rad=0.",
							  shrinkA=0, shrinkB=10,
							  arrowstyle= '-|>', ls= '-', linewidth=1
							  ),
			  va='bottom', ha='left', zorder=19
			  )
		#ax2.text(xPos + .01 * (x1 - x0), yPos,
		#    '({:.2f}, {:.2f})'.format(x,y),
		#    transform=ax.transData, va='center')

	#ax2.annotate('peak', xy = (features[result[0][0]][0], features[result[0][0]][1]))
	#ax2.annotate('rnn', xy = (rnn_last[0], rnn_last[1]))
	#ax2.annotate('max', xy = (max_feature[0], max_feature[1]))
	#ax2.annotate('avg', xy = (avg_feature[0], avg_feature[1]))
	#ax2.scatter(features[result[0][0]][0], features[result[0][0]][1], color='C6', marker='p')
	#ax2.scatter(rnn_last[0], rnn_last[1], color='C7', marker='o')
	#ax2.scatter(max_feature[0], max_feature[1], color='C8', marker='d')
	#ax2.scatter(avg_feature[0], avg_feature[1], color='C9', marker='D')
	ax2.legend(['0', '1', '2', '3', '4', '5', 'p', 'r', 'm', 'a'], loc='best')

	#path = "MILExperiments"
	#dirname = osp.join(path, subject)
	#if not osp.exists(dirname):
	#    os.mkdir(dirname)
	#irname2 = osp.join(dirname, prefix)
	#f not osp.exists(dirname2):
	#    os.mkdir(dirname2)
	#    "/" + "Epoch"  +str(epoch)  +"/"+ str(batch_idx) + "_" + str(imagelabels[time].data.cpu())
	plt.savefig(save_name, bbox_inches='tight')
	#plt.savefig(save_name)
	plt.close(fig)
	#sys.exit()

def init_params(net):
	'''Init layer parameters.'''
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
			if m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif isinstance(m, nn.BatchNorm2d):
			init.normal_(m.weight.data, 1.0, 0.02)
			init.constant_(m.bias.data, 0.0)
		elif isinstance(m, nn.Linear):
			init.normal_(m.weight.data, std=1e-2)
			init.constant_(m.bias.data, 0.0)
	print("initialization done")


def online_mean_and_sd(loader):
	"""Compute the mean and sd in an online fashion

		Var[x] = E[X^2] - E^2[X]
	"""
	cnt = 0
	fst_moment = torch.empty(3)
	snd_moment = torch.empty(3)

	#mean = 0.
	#std = 0.
	#nb_samples = 0.

	for batch_idx, (data, target) in enumerate(loader):
		b, c, s, h, w = data.size()
		# batch_samples = data.size(0)
		#data = data.view(b, s, c, -1)
		#mean += data.mean(3).sum(0)
		#std += data.std(3).sum(0)
		#nb_samples += b

		nb_pixels = b * h * w * s
		sum_ = torch.sum(data, dim=[0, 2, 3, 4])
		sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3, 4])
		fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
		snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
		cnt += nb_pixels

	#mean /= (nb_samples * s)
	#std /= (nb_samples * s)
	#print(mean)
	#print(std)
	#sys.exit()

	return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def online_mean_and_sd_withoutbatch(loader, root):
	"""Compute the mean and sd in an online fashion

		Var[x] = E[X^2] - E^2[X]
	"""
	cnt = 0
	fst_moment = torch.empty(3)
	snd_moment = torch.empty(3)

	# mean = 0.
	# std = 0.
	# nb_samples = 0.
	cnt =0
	sequences = []
	for seq in loader:
		sequence = []
		for img in seq:
			imgPath = img[0]
			img = cv2.imread(root + imgPath)
			w,h,c = img.shape
			if w == 0:
				continue
			else:
				img = cv2.resize(img, (224, 224))[:, :, [2, 1, 0]]
				sequence.append(img)
		sequences.append(sequence)
		cnt = cnt + 1
		#if (cnt == 10):
		#    break
	np_sequences = np.asarray(sequences)

	batch_mean = np.mean(np_sequences, axis=(0,1,2,3))
	batch_std0 = np.std(np_sequences, axis=(0,1,2,3))
	print(batch_mean)
	print(batch_std0)
	sys.exit()
	# mean /= (nb_samples * s)
	# std /= (nb_samples * s)
	# print(mean)
	# print(std)
	# sys.exit()

	return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

class VisdomLinePlotter(object):
	"""Plots to Visdom"""
	def __init__(self, env_name='main',port=8097):
		self.viz = Visdom(port=port)
		self.env = env_name
		self.plots = {}
	def plot(self, var_name, split_name, title_name, x, y):
		if var_name not in self.plots:
			self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
				legend=[split_name],
				title=title_name,
				xlabel='Epochs',
				ylabel=var_name
			))
		else:
			#self.viz.text("Hello world")
			self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

def add_weight_decay(net, l2_value):
	decay, no_decay = [], []
	for name, param in net.named_parameters():
		if not param.requires_grad:
			print("Freezed")
			continue  # frozen weights
		if len(param.shape) == 1 or name.endswith(".bias"):
			no_decay.append(param)
			print("weight decay rejected on bias")
		else:
			decay.append(param)
			print("weight decay applied on bias")
	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

def MinmaxNormalize(data):
	maxVal = np.nanmax(data)
	minVal = np.nanmin(data)
	return [(x-minVal)/(maxVal-minVal) for x in data]

def MeanNormalization (ref, pred):
	mean_ref = np.mean (ref)
	mean_pred = np.mean (pred)
	bias = mean_ref - mean_pred
	return pred + bias

def GetStdRatio(labels, pred):
	label_sd = np.std(labels)
	pred_sd = np.std(pred)
	sd = label_sd / pred_sd
	return sd

def ApplyScaling (pred, ratio):
	return [x*ratio for x in pred]

def MedianFilter (data, windowLength):
	return signal.medfilt (data, windowLength)

def Normalize (pred, devLabels):
	# std prediction normalization
	#predTrain = reg.predict (trainData)

	#devLabels = devLabels
	#pred = pred

	#filt_devLabels = MedianFilter (devLabels, 3)
	#filt_pred = MedianFilter (pred, 3)

	if (np.nanmax(devLabels) - np.nanmin(devLabels)) == 0:
		print(devLabels)
		tar = devLabels
	else:
		tar = MinmaxNormalize(devLabels)

	if (np.nanmax(pred) - np.nanmin(pred)) == 0:
		print(pred)
	else:
		pred = MinmaxNormalize(pred)

	# pred = MeanNormalization (devLabels, pred)
	# ratio = GetStdRatio (devLabels, pred)
	# pred = ApplyScaling (pred, ratio)

	# Mean bias normalization
	#pred = MeanNormalization(tar, pred)
	return pred, tar

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= [] 
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])