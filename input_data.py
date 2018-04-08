from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.misc
from glob import glob
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def input_pipeline_train(data_dir, batch_size):
	fn_color = sorted(glob(data_dir + '/color/*.png'))
	fn_mask = sorted(glob(data_dir + '/mask/*.png'))
	fn_normal = sorted(glob(data_dir + '/normal/*.png'))

	reader = tf.WholeFileReader()

	fn_color_queue = tf.train.string_input_producer(fn_color, shuffle=False)
	_, value = reader.read(fn_color_queue)
	color = tf.image.decode_png(value, channels=3)

	fn_mask_queue = tf.train.string_input_producer(fn_mask, shuffle=False)
	_, value = reader.read(fn_mask_queue)
	mask = tf.image.decode_png(value, channels=1)

	fn_normal_queue = tf.train.string_input_producer(fn_normal, shuffle=False)
	_, value = reader.read(fn_normal_queue)
	normal = tf.image.decode_png(value, channels=3)
	x = color
	normal.set_shape([128, 128, 3])
	mask.set_shape([128, 128, 1])
	x.set_shape([128, 128, 3])

	x, mask, normal = tf.train.shuffle_batch([x, mask, normal],
					batch_size=batch_size,
					num_threads=4,
					capacity=1000 + 3 * batch_size,
					min_after_dequeue=1000)
	x = tf.cast(x, tf.float32)
	normal = tf.cast(normal, tf.float32)
	mask = tf.cast(mask, tf.float32)
	return x, mask, normal

def input_pipeline_test(data_dir, batch_size):
	fn_color = glob(data_dir + '/color/*.png')
	fn_mask = sorted(glob(data_dir + '/mask/*.png'))
	fn_color = natural_sort(fn_color)
	fn_mask = natural_sort(fn_mask)
	reader = tf.WholeFileReader()

	fn_color_queue = tf.train.string_input_producer(fn_color, shuffle=False)
	_, value = reader.read(fn_color_queue)
	color = tf.image.decode_png(value, channels=3)

	fn_mask_queue = tf.train.string_input_producer(fn_mask, shuffle=False)
	_, value = reader.read(fn_mask_queue)
	mask = tf.image.decode_png(value, channels=1)

	x = color
	mask.set_shape([128, 128, 1])
	x.set_shape([128, 128, 3])

	# x = color * mask
	x, mask = tf.train.batch([x, mask],
					batch_size=batch_size,
					num_threads=4,
					capacity=10000 + 3 * batch_size,
					enqueue_many=False,
					dynamic_pad=True)


	x = tf.cast(x, tf.float32)
	mask = tf.cast(mask, tf.float32)
	return x, mask
