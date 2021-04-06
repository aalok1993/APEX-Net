import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, Model
import os
from utils import *
from model import *
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

if __name__ == '__main__':

	train_data_path = 'data/train/'
	test_data_path  = 'data/test/'
	batch_size      = 16
	K = 10 # Number of maximum functions we are assuming
	input_shape = (256, 256, 3)

	model = CountNet(K, input_shape=input_shape)
	optimizer_method = optimizers.Adam(lr=0.0001)

	if not os.path.isdir('checkpoint'):
		os.makedirs('checkpoint')
	if not os.path.isdir('results'):
		os.makedirs('results/train')
		os.makedirs('results/val')
	ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer_method, model=model)
	manager = tf.train.CheckpointManager(ckpt, directory='checkpoint/', max_to_keep=10)
	ckpt.restore(manager.latest_checkpoint)
	if manager.latest_checkpoint:
		print('\n\nRestored from last checkpoint : {0} \n\n'.format(int(ckpt.step)))

	EPOCHS = 1000
	START  = 0
	model_save_freq = 1000
	vis_freq = 100

	for epoch in range(START, EPOCHS):
		
		total_train_loss = metrics.Mean()
		total_val_loss   = metrics.Mean()

		train_batch = load_data(train_data_path, batch_size=batch_size)

		for itr, (X_tf, Y_tf) in enumerate(tqdm(train_batch)):
			itr_train_loss = metrics.Mean()
			X, Y = fetch_value(X_tf, Y_tf, batch_size, K)
			loss = train_step(model, optimizer_method, X, Y)
			total_train_loss.update_state(loss)
			itr_train_loss.update_state(loss)
			ckpt.step.assign_add(1)
			if (itr%model_save_freq)==0:
				manager.save()
			if (itr%vis_freq)==0:
				vis_result(int(ckpt.step), 'results/train',model, K, X_tf, Y_tf)
			print('Iter :{0}\tIter Train Loss :{1}'.format(itr, itr_train_loss.result()))


		val_batch   = load_data(test_data_path, batch_size=batch_size)

		for itr, (X_tf, Y_tf) in enumerate(tqdm(val_batch)):
			itr_val_loss = metrics.Mean()
			X, Y = fetch_value(X_tf, Y_tf, batch_size, K)
			loss = val_step(model, X, Y)
			total_val_loss.update_state(loss)
			itr_val_loss.update_state(loss)
			if (itr%vis_freq)==0:
				vis_result(int(ckpt.step)*(itr+1), 'results/val',model, K, X_tf, Y_tf)
			print('Iter :{0}\tIter Val Loss :{1}'.format(itr, itr_val_loss.result()))

		ckpt.step.assign_add(1)
		manager.save()

		print('\nEpoch :{0}\tTotal Train Loss :{1}\tTotal Val Loss : {2}\n'.format(epoch, total_train_loss.result(), total_val_loss.result()))
		with open('log.txt', 'a') as file:
			file.write('\nEpoch :{0}\tTotal Train Loss :{1}\tTotal Val Loss : {2}\n'.format(epoch, total_train_loss.result(), total_val_loss.result()))