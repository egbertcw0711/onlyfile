from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from glob import glob
from model import *
from PIL import Image


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

	for k in range(20000):
		sess.run(train_step)

        print('%d. Loss_1: %.6f   Loss_2: %.6f' % (k, sess.run(loss_1), sess.run(loss_2)))

		if (k + 1) % 10 == 0:
			saver.save(sess, './checkpoints/model.ckpt', global_step=k + 1)
		if (k + 1) % 500 == 0:
			rate *= 0.3
			optim = tf.train.AdamOptimizer(learning_rate=rate)
                
    ###---------------Below is picture save part!---------------------
    selected_dir = ['train/prediction_selected', 'train/mask_selected', 'train/normal_selected']

    for dir in selected_dir:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        
        os.makedirs(dir)

    batch_index = random.sample(range(0, 20000), 50)    # create random number to select the batch randomly

    #Save the images
    for index in range(50):
        result = Image.fromarray((pred[index, :, :, :]).astype(np.uint8))
        #print("result.shape ==", result.get_shape())
        result.save('train/prediction_selected/' + str(batch_index[index]) + '.png')
        
        mask_selected = Image.open('train/mask/' + str(batch_index[index]) + '.png')
        mask_selected.save('train/mask_selected/' + str(batch_index[index]) + '.png')
        
        mask_selected = Image.open('train/normal/' + str(batch_index[index]) + '.png')
        mask_selected.save('train/normal_selected/' + str(batch_index[index]) + '.png')


    #Compute MAE

    prediction_folder = './train/prediction_selected'
    normal_folder = './train/normal_selected'
    mask_folder = './train/mask_selected'
    mae = Evaluation_script_.evaluate(prediction_folder, normal_folder, mask_folder,)
    print("Final MAE ==", mae)
        
        
	coord.request_stop()
	coord.join(threads)



	

