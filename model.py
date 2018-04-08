from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from glob import glob
from input_data import *


def build_model(phase, batch_size, flag):
	if (flag == 'train'):
		x, mask, y_ = input_pipeline_train('./' + phase, batch_size)

	if (flag == 'test'):
		x, mask = input_pipeline_test('./' + phase, batch_size)
	x = tf.concat([x, mask], axis = 3)
	# print(x)
	h_conv1 = tf.layers.conv2d(x, 20, 5, 1, padding='same', name='conv1') #bs*128*128*64 (batch_size)
	h_pool1 = tf.layers.max_pooling2d(tf.nn.relu(h_conv1), 2, 2, padding='same', name='pool1')
	h_conv11 = tf.layers.conv2d(x, 10, 5, 1, padding='same', name='conv11')
	h_conv2 = tf.layers.conv2d(h_pool1, 30, 10, 1, padding='same', name='conv2') #bs*128*128*64 (batch_size)
	h_pool2 = tf.layers.max_pooling2d(tf.nn.relu(h_conv2), 2, 2, padding='same', name='pool2')
	h_conv22 = tf.layers.conv2d(h_conv2, 10, 5, 1, padding = 'same', name='conv22')
	h_conv3 = tf.layers.conv2d(h_pool2, 5, 20, 1, padding='same', name='conv3') #bs*128*128*64 (batch_size)
	h_pool3 = tf.layers.max_pooling2d(tf.nn.relu(h_conv3), 2, 2, padding='same', name='pool3')
	h_conv4 = tf.layers.conv2d(h_pool3, 5, 25, 1, padding='same', name='conv4') #bs*128*128*64 (batch_size)
	h_conv5 = tf.layers.conv2d(tf.nn.relu(h_conv4), 3, 30, 1, padding='same', name='conv5') #bs*128*128*64 (batch_size)

	h_conv5 = tf.concat([h_conv4, h_conv5], axis=3)
	h_transpose_4 = tf.layers.conv2d_transpose(h_conv5, 16, 4, 1, padding='same', name='h_transpose_3')
	h_concat3 = tf.concat([h_transpose_4, h_conv4], axis=3)
	h_transpose_3 = tf.layers.conv2d_transpose(h_concat3, 8, 4, 2, padding='same', name='h_transpose_2') #
	h_concat2 = tf.concat([h_transpose_3, h_conv3], axis=3)
	h_transpose_2 = tf.layers.conv2d_transpose(h_transpose_3, 5, 4, 2, padding='same', name='h_transpose_1')
	h_concat1 = tf.concat([h_transpose_2, h_conv22], axis=3)
	h_transpose_1 = tf.layers.conv2d_transpose(h_concat1, 3, 4, 2, padding='same', name='h_transpose_0')

	h_concat = tf.concat([h_transpose_1, h_conv11, x], axis=3)
	pred = tf.layers.conv2d(h_concat, 3, 5, 1, padding='same', name='conv_pred')
	print(pred)
	if (flag == 'test'):
		return pred

	mask = tf.concat([mask, mask, mask], axis=3)
	mask_region = tf.not_equal(mask,tf.zeros_like(mask))
	loss_with_mask = tf.reduce_mean(tf.boolean_mask(tf.abs(pred - y_),mask_region))
	# loss_with_mask = tf.reduce_mean(tf.abs(pred - y_))

	loss = tf.reduce_mean(tf.abs(pred - y_))

	return loss, loss_with_mask, pred



