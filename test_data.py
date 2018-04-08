# import tensorflow as tf
# from LearningTest import build_model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model import *
import scipy.misc

import tensorflow as tf
from glob import glob
from matplotlib import pyplot as plt
import argparse
# import model
import re
import os

def run(checkpoints_folder, test_image_folder, test_size, checkpoints_idx, output_folder):
	test_size = int(test_size)
	phase = test_image_folder
	batch_size = int(test_size)
	divid = 500
	each_batch = int(batch_size / 500)
	pred = build_model(phase, int(batch_size / 500), 'test')
	pictures = []

	if os.path.isdir('./' + output_folder):
		pass
	else:
		os.mkdir('./' + output_folder)

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		new_saver = tf.train.Saver()
		new_saver.restore(sess, './' + checkpoints_folder + '/model.ckpt-' + str(checkpoints_idx))

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		
		for i in range(divid):
			# for j in range(each_batch):
			pictures = sess.run(pred)
			for j in range(each_batch):
				scipy.misc.imsave('./' + output_folder + '/' + str(j + i * each_batch) + '.png', pictures[j])


		coord.request_stop()
		coord.join(threads)


	# print(pictures)



if (__name__ == '__main__'):
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoints_folder', '-c', required=True, help='Folder path for checkpoints')
	parser.add_argument('--test_image_folder', '-t', required=True, help='Folder path for test images')
	parser.add_argument('--test_size', '-s', required=True, help='number for test images size')
	parser.add_argument('--checkpoints_idx', '-i', required=True, help='number for index of training')
	parser.add_argument('--output_folder', '-o', required=True, help='output folder for result')

	args = parser.parse_args()
	# print(args)
	run(**vars(args))












