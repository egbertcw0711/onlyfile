from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from glob import glob
from model import *

phase='train'
batch_size = 80


loss_1, loss_2, pred = build_model(phase, batch_size, 'train')

rate = 0.0001
optim = tf.train.AdamOptimizer(learning_rate=rate)
train_step = optim.minimize(loss_1)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
    
	sess.run(init)

	saver = tf.train.Saver(max_to_keep=10)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for k in range(200000):
		sess.run(train_step)

		print('%d. Loss_1: %.6f   Loss_2: %.6f' % (k, sess.run(loss_1), sess.run(loss_2)))

		if (k + 1) % 10 == 0:
			saver.save(sess, './checkpoints/model.ckpt', global_step=k + 1)
		if (k + 1) % 500 == 0:
			rate *= 0.3
			optim = tf.train.AdamOptimizer(learning_rate=rate)
	coord.request_stop()
	coord.join(threads)


# if (__main__ == '__main__'):

	

