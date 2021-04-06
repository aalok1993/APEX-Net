import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, Model
import os
from loss_methods import *
import matplotlib.pyplot as plt
import glob

input_shape = (256, 256)

def load_data(data_path, batch_size=16):
	# Here we are loading training data
	X = glob.glob(data_path + '*.jpg')
	Y = [data_path+os.path.splitext(os.path.basename(x))[0]+'.npy' for x in X]
	batch = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(len(X)).batch(batch_size, drop_remainder=True)
	return batch

def get_img(img_path):
	img_path = img_path.numpy().decode('utf-8')
	img = cv2.imread(img_path, 1)
	img = cv2.resize(img, input_shape, interpolation=cv2.INTER_CUBIC)
	# Here we are normalizing image
	img = (tf.cast(img, dtype=tf.float32) - 127.5) / 127.5
	return img

def get_nparr(np_path, A):
	np_path = np_path.numpy().decode('utf-8')
	nparr = np.load(np_path)
	nonzero = nparr.shape[0]
	zero    = A - nparr.shape[0]
	h_fill  = np.zeros((A - nparr.shape[0], 1024))
	v_fill  = np.vstack([np.ones((nonzero, 1)), np.zeros((zero, 1))])
	nparr = np.hstack([v_fill, np.vstack([nparr, h_fill])])
	return tf.cast(tf.convert_to_tensor(nparr), dtype=tf.float32)

def fetch_value(X, Y, batch_size=16, A=10):
	X = tf.convert_to_tensor(list(map(get_img, X)))
	Y = list(map(get_nparr, Y, [A]*batch_size))
	Y = tf.convert_to_tensor(Y)
	return X, Y

def vis_result(epoch, figpath, model, K, X, Y):
	X, Y = fetch_value(X, Y, 1, K)
	Y_cap_plot, Y_cap_prob = model(X, training=False)

	x =np.linspace(0,1,1024)
	c = ['b','g','r','c','m','y','k','b','g','r']

	X = X[0]
	Y = Y[0]
	Y_cap_plot = Y_cap_plot[0]
	Y_cap_prob = Y_cap_prob[0]

	total_plot = np.sum(Y[:, 0]==1)
	N = 10
	M = 10

	plt.figure(figsize=(10,10))
	ax = plt.subplot2grid((N, M), (0, 0), rowspan=5, colspan=4)
	ax.set_title('Input Plots\nTF:{0}'.format(total_plot), y=-0.01)
	ax.axis('off')
	ax.imshow(X[:,:,::-1].numpy()*0.5+0.5)

	plot_pos = 0
	for i in range(0, 2*total_plot, 2):
		for j in range(4, 10, 2):
			if total_plot<=plot_pos:
				break
			ax = plt.subplot2grid((N, M), (i, j), rowspan=2, colspan=2)
			ax.set_title('IP-Fn:{0}'.format(plot_pos+1), y=-0.01)
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.set_ylim(0,1)
			ax.set_xlim(0,1)
			ax.axis('off')
			ax.plot(x,Y[plot_pos, 1:],c=c[plot_pos])
			plot_pos += 1
		if total_plot<=plot_pos:
			break

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
	
	plt.savefig('{0}/{1}.png'.format(figpath, epoch))
	plt.close('all')

# @tf.function
def train_step(model, optimizer_method, X, Y):
	with tf.GradientTape() as tape:
		Y_cap_plot, Y_cap_prob = model(X, training=True)
		plot_loss, prob_loss  = cross_n_prob_loss(Y, Y_cap_plot, Y_cap_prob)
		total_loss =  plot_loss + prob_loss

	gradients = tape.gradient(total_loss, model.trainable_variables)
	optimizer_method.apply_gradients(zip(gradients, model.trainable_variables))
	return total_loss

# @tf.function
def val_step(model, X, Y):
	with tf.GradientTape() as tape:
		Y_cap_plot, Y_cap_prob = model(X, training=True)
		plot_loss, prob_loss  = cross_n_prob_loss(Y, Y_cap_plot, Y_cap_prob)
	total_loss = prob_loss + plot_loss
	return total_loss