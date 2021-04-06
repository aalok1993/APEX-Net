import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, Model
import os
from model import CountNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import pdb

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

def load_data(data_path, batch_size=16):
	X = []
	# Here we are loading training data
	for filename in sorted(os.listdir(data_path)):
		# Here we are joining filename with relative path
		path = os.path.join(data_path, filename)
		# Here we are extracting extension from file path
		file_ext  = os.path.splitext(path)[1]
		if file_ext != '.npy':
			X.append(path)

	batch = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)

	return batch

def get_img(img_path):
	img_path = img_path.numpy().decode('utf-8')
	img = cv2.imread(img_path, 1)
	img = cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
	# Here we are normalizing image
	img = (tf.cast(img, dtype=tf.float32) - 127.5) / 127.5
	return img

def fetch_value(X, batch_size=1, A=10):
	X = tf.convert_to_tensor(list(map(get_img, X)))
	return X

def vis_result(epoch, figpath, model, K, X):
	X = fetch_value(X, 1, K)
	Y_cap_plot, Y_cap_prob = model(X, training=False)

	x =np.linspace(0,1,1024)
	c = ['b','g','r','c','m','y','k','b','g','r']

	X = X[0]
	Y_cap_plot = Y_cap_plot[0]
	Y_cap_prob = Y_cap_prob[0]

	N = 10
	M = 10

	plt.figure(figsize=(10,10))

	ax = plt.subplot2grid((N, M), (5, 0), rowspan=5, colspan=4)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax.set_ylim(0,1)
	ax.set_xlim(0,1)
	ax.axis('off')
	total_fn = 0
	pos_list = []
	for i in range(Y_cap_prob.shape[-1]):
		if Y_cap_prob[i]>=0.5: 
			pos_list.append(i)
			ax.plot(x,Y_cap_plot[i],c=c[i])
			total_fn += 1
	ax.set_title('Predicted Plots\nTF:{0}'.format(total_fn), y=-0.01)
	plot_pos = 0
	for i in range(5, 5+2*total_fn, 2):
		for j in range(4, 10, 2):
			if total_fn<=plot_pos:
				break
			ax = plt.subplot2grid((N, M), (i, j), rowspan=2, colspan=2)
			ax.set_title('OP-Fn:{0}'.format(plot_pos+1), y=-0.01)
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.set_ylim(0,1)
			ax.set_xlim(0,1)
			ax.axis('off')
			ax.plot(x,Y_cap_plot[pos_list[plot_pos]],c=c[pos_list[plot_pos]])
			plot_pos += 1
		if total_fn<=plot_pos:
			break
	
	# plt.show()
	plt.savefig('{0}/{1}.png'.format(figpath, epoch))


if __name__ == '__main__':

	# train_data_path = 'data/train/'
	test_data_path  = 'unseen_data/'
	batch_size      = 1
	K = 10 # Number of maximum functions we are assuming
	val_batch   = load_data(test_data_path, batch_size=batch_size)
	model = CountNet(K, input_shape=(256, 256, 3))
	if not os.path.isdir('results/unseen'): os.makedirs('results/unseen')

	ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=model)
	manager = tf.train.CheckpointManager(ckpt, directory='checkpoint/', max_to_keep=100)
	ckpt.restore(manager.latest_checkpoint)
	# pdb.set_trace()
	if manager.latest_checkpoint:
		print('Restored from last checkpoint : {0}'.format(int(ckpt.step)))

	# EPOCHS = 10000
	# START  = 0
	# model_save_freq = 1000
	# vis_freq = 100

	for itr, X_tf in enumerate(tqdm(val_batch)):
		vis_result(itr, 'results/unseen',model, K, X_tf)
