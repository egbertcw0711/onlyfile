import tensorflow as tf
import imageio
import os
import numpy as np
import random
from PIL import Image
import glob
import random
# import scipy

def readimage(folder, index):
    path = os.path.join(folder, str(index)+'.png')
    image = imageio.imread(path)
    return image

def readmask(folder, index):
    path = os.path.join(folder, str(index)+'.png')
    image = imageio.imread(path)
    return image

def readnormal(folder, index):
    path = os.path.join(folder, str(index)+'.png')
    image = imageio.imread(path)
    return image

def normalize(x):
    # print(x.shape)
    res = np.zeros(x.shape)
    for i in range(3):
        max_val = np.max(x[:,:,i])
        min_val = np.min(x[:,:,i])
        res[:,:,i] = 1.0*(x[:,:,i] - min_val)/(max_val-min_val+1)
    return res

batch_size = 20
save_model_path = './best_model/surface_normal_est'

test_color = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')
test_mask = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')


# build the graph
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:

	loader = tf.train.import_meta_graph(save_model_path+'.meta')
	loader.restore(sess, save_model_path)

	loaded_x = loaded_graph.get_tensor_by_name('x:0')
	loaded_y = loaded_graph.get_tensor_by_name('y:0')
	loaded_z = loaded_graph.get_tensor_by_name('z:0')
	loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
	loaded_output = loaded_graph.get_tensor_by_name('output:0')

	num_test = len(glob.glob('./test/color/*.png'))
	batches = num_test // batch_size
	for i in range(batches):
		print('{:d}/{:d}'.format(i,batches))
		for j in range(batch_size):
			# print(j)
			test_color[j,:,:,:] = readimage('./test/color',j + i*batch_size)
			test_mask[j,:,:,0] = readmask('./test/mask', j + i*batch_size)
			test_mask[j,:,:,1] = readmask('./test/mask', j + i*batch_size)
			test_mask[j,:,:,2] = readmask('./test/mask', j + i*batch_size)
			test_color[counter,:,:,:] /= np.amax(test_color[counter,:,:,:]+1)
			test_mask[counter,:,:,:] /= np.amax(test_mask[counter,:,:,:]+1)
		
		predictions = sess.run(loaded_output, feed_dict={loaded_x:test_color,loaded_y:test_mask,loaded_keep_prob:1.0})
		
		for k in range(batch_size):
			image=Image.fromarray((255.0*predictions[k,:,:,:]).astype(np.uint8))
			print(k+i*batch_size)
			image.save('./test/normal/'+str(k+i*batch_size)+'.png')