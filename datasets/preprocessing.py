import pandas as pd
import os
import sys
anno_path = "/export/livia/home/vision/pgan/Datasets/Affwild2/annotations"
img_path = "/export/livia/home/vision/pgan/Datasets/Affwild2/cropped_aligned"


def produce_multi_task_videos():
	train_videos = os.listdir(os.path.join(anno_path, "VA_Set", "Train_Set"))
	val_videos = os.listdir(os.path.join(anno_path, "VA_Set", "Validation_Set"))
	train_videos = list(set(train_videos))
	val_videos = list(set(val_videos))
	return train_videos,val_videos

train_videos,val_videos = produce_multi_task_videos()

def get_names(id):
	name = ""
	if id>=0 and id<10:
		name = "0000" + str(id)
	elif id>=10 and id<100:
		name = "000" + str(id)
	elif id>=100 and id<1000:
		name = "00" + str(id)
	elif id>=1000 and id<10000:
		name = "0" + str(id)
	else:
		name = str(id)
	return name

def produce_multi_task_labels_for_one_video(video_name, flag):
	label_dict = {}
	VA_flag = True
	#if video_name+".txt" not in val_videos:
	#	VA_flag = False
	#if not VA_flag:
	#	return
	if VA_flag:
		f = open(os.path.join(anno_path,"VA_Set","Validation_Set",video_name+".txt"))
		lines = f.readlines()[1:]
		for i in range(len(lines)):
			l = lines[i].strip().split(",")
			if l[0] == "-5" or l[1] == "-5":
				print(l)
				continue
			frame = get_names(i+1)
			#if os.path.exists(os.path.join(img_path,video_name,frame+".jpg")):
			n = video_name.split(".")[0]+"/"+frame+".jpg"
			if n not in label_dict.keys():
				label_dict[n] = [float(l[0]),float(l[1]), frame]
			else:
				print(n)
				sys.exit()
				label_dict[n][0] = float(l[0])
				label_dict[n][1] = float(l[1])
				label_dict[n][2] = float(frame)
	return label_dict


def produce_anno_csvs(videos, flag):
	save_path = "/export/livia/home/vision/pgan/Datasets/Affwild2/annotations/preprocessed_all_labeled_images_new/" + flag
	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	for video in videos:
		label_dict = produce_multi_task_labels_for_one_video(video.split(".")[0], flag)
		print(len(label_dict))
		data = pd.DataFrame()
		imgs,V,A,frame_id = [],[],[],[]
		for k,v in label_dict.items():
			imgs.append(k)
			V.append(v[0])
			A.append(v[1])
			frame_id.append(v[2])
		data["img"],data["V"],data["A"], data["frame_id"] = imgs,V,A, frame_id
		data.to_csv(os.path.join(save_path,video.split(".")[0]+".csv"))

def produce_total_csvs(flag):
	path = "/export/livia/home/vision/pgan/Datasets/Affwild2/annotations/preprocessed_all_labeled_images/" + flag
	csv_data_list = os.listdir(path)
	total_data = pd.DataFrame()
	imgs,V,A = [],[],[]
	for csv in csv_data_list:
		print(csv)
		data = pd.read_csv(os.path.join(path,csv))
		imgs.extend(data["img"].to_list())
		V.extend(data["V"].to_list())
		A.extend(data["A"].to_list())
	print(len(imgs),len(A),len(V))
	total_data["img"],total_data["V"],total_data["A"] = imgs,V,A
	total_data.to_csv("ABAW2_multi_task_training.csv")

def produce_category_csvs():
	path = "/export/livia/home/vision/pgan/Datasets/Affwild2/annotations/preprocessed_AV/Train_Set"
	csv_data_list = os.listdir(path)
	VA_spec_data = []
	multi_data = []
	for csv in csv_data_list:
		print(csv)
		total_data = pd.read_csv(os.path.join(path,csv))
		imgs, V, A = total_data["img"],\
						  total_data["V"],total_data["A"]
		for i in range(len(imgs)):
			if V[i] != -1 and A[i] != -1:
				 multi_data.append(total_data.iloc[i, :])
		print(len(multi_data))
		print(len(VA_spec_data))
	multi_data = pd.DataFrame(multi_data)

	multi_data.to_csv("multi_train_data.csv")

def produce_training_data():
	total_data = pd.read_csv("ABAW2_multi_task_training.csv")
	imgs, AU, EXP, V, A =  total_data["img"], total_data["AU"], total_data["EXP"], total_data["V"], total_data["A"]

	for i in range(len(imgs)):
		if i%1000==0:
			print(i)
		if AU[i]==-1 or V[i]==-1 or A[i]==-1 or EXP[i]==-1:
			total_data.drop([i])

	print(len(total_data))

#produce_training_data()
#produce_category_csvs()
#produce_total_csvs()
count = 0
#for videos in [train_videos, val_videos]:
#    flag = ['Train_Set', 'Val_Set']
produce_anno_csvs(val_videos, 'Val_Set')