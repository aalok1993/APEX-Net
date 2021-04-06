import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, Model
import os
import pdb

def cross_n_prob_loss(GT, PRED_PLOT, PRED_PROB):
	total_plot_error = 0
	total_prob_error = 0
	count = 0
	for Y, Y_pred, Y_prob_pred in zip(GT, PRED_PLOT, PRED_PROB):
		Y = Y[Y[:, 0]==1][:, 1:]
		M = Y.shape[-2]
		N = Y_pred.shape[-2]
		Y = tf.expand_dims(Y, axis=-2)
		Y_pred = tf.expand_dims(Y_pred, axis=-2)
		new_Y = tf.tile(Y, (1, N, 1))
		new_Y_pred = tf.tile(Y_pred, (1, M, 1))
		new_Y_pred = tf.transpose(new_Y_pred, perm=[1,0,2])
		chamf_dis = tf.norm(tf.subtract(new_Y, new_Y_pred), axis=-1)
		minval_sum = tf.reduce_sum(tf.reduce_min(chamf_dis, axis=-1))
		minval_idx = tf.argmin(chamf_dis, axis=-1)
		total_plot_error += minval_sum

		Y_prob = tf.zeros(N,dtype=tf.float32)
		minval_idx,_ = tf.unique(minval_idx)
		Y_prob = tf.tensor_scatter_nd_update(Y_prob, tf.expand_dims(minval_idx,-1), tf.ones_like(minval_idx,dtype=tf.float32))
		
		total_prob_error += losses.BinaryCrossentropy()(Y_prob, Y_prob_pred)
		count+=1
	
	return total_plot_error/count, total_prob_error/count