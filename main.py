# from __future__ import print_function
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
    res = np.zeros(x.shape)
    for i in range(3):
        max_val = np.amax(x[:,:,i])
        min_val = np.amin(x[:,:,i])
        res[:,:,i] = 1.0*(x[:,:,i]-min_val)/(max_val-min_val+1.0)
    return res

def weight_variable(shape):
    """
    init the weight variables with truncated normal distribution
    """
    initial = tf.truncated_normal(shape,stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    init the bias term with constant
    """
    initial = tf.constant(0.0,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    """
    perfrom 2-d convolution
    
    inputs: x (h x w x in_depth)
    weights: w (kernel_size x kernel_size x in_depth x num_filters)
    
    return: output(h x w x num_filters)
    """
    return tf.nn.conv2d(x,w,strides=[1,1,1,1], padding='SAME')

def hourglass(inputs, in_dim, out_dim, inter_dim, conv1_size, conv2_size, conv3_size, conv4_size, keep_prob):
    """
    a particular layer of the hourglass architecture mensioned in the papaer
    inputs: example h x w x 128
    in_dim: input dimsion
    out_dim: output dimsion
    inter_dim: interim dimsion
    conv1_size: kernel size of the convolution layer 1, usually is square
    conv2_size: kernel size of the convolution layer 2
    conv3_size, conv4_size: the same as above
    The 1x1 conv before entering into the conv2, conv3, conv4 as mensioned in the papaer
    """
    w02 = weight_variable([1,1,in_dim,inter_dim])
    b02 = bias_variable([inter_dim])
    conv02 = tf.nn.relu(conv2d(inputs, w02) + b02)
  

    w03 = weight_variable([1,1,in_dim,inter_dim])
    b03 = bias_variable([inter_dim])
    conv03 = tf.nn.relu(conv2d(inputs, w03) + b03)


    w04 = weight_variable([1,1,in_dim,inter_dim])
    b04 = bias_variable([inter_dim])
    conv04 = tf.nn.relu(conv2d(inputs, w04) + b04)


    w1 = weight_variable([conv1_size,conv1_size,in_dim,out_dim//4])
    b1 = bias_variable([out_dim//4])
    conv1 = tf.nn.relu(conv2d(inputs,w1) + b1)   
    
    w2 = weight_variable([conv2_size,conv2_size,inter_dim,out_dim//4])
    b2 = bias_variable([out_dim//4])
    conv2 = tf.nn.relu(conv2d(conv02,w2) + b2)
    
    w3 = weight_variable([conv3_size,conv3_size,inter_dim, out_dim//4])
    b3 = bias_variable([out_dim//4])
    conv3 = tf.nn.relu(conv2d(conv03,w3) + b3)
    
    w4 = weight_variable([conv4_size,conv4_size,inter_dim, out_dim//4])
    b4 = bias_variable([out_dim//4])
    conv4 = tf.nn.relu(conv2d(conv04,w4) + b4)
    
    res = tf.concat([conv1, conv2, conv3, conv4], axis=3)
    
    return res


def train_test_split(random_indexes,validation_size):
    train_indexes = random_indexes[:len(random_indexes)-validation_size]
    test_indexes = random_indexes[len(random_indexes)-validation_size:]
    return train_indexes, test_indexes

def get_batches(random_indexes, batch_size):
    num_batches = len(random_indexes) // batch_size
    indexes = random_indexes[:num_batches*batch_size]
    for idx in range(0, len(indexes),batch_size):
        yield indexes[idx:idx+batch_size]

def scan_png_files(folder):
    '''
    folder: 1.png 3.png 4.png 6.png 7.exr unknown.mpeg
    return: ['1.png', '3.png', '4.png']
    '''
    ext = '.png'
    ret = [fname for fname in os.listdir(folder) if fname.endswith(ext)]

    return ret


def evaluate(prediction_folder, groundtruth_folder, mask_folder):
    '''
    Evaluate mean angle error of predictions in the prediction folder,
    given the groundtruth and mask images.
    '''
    # Scan folders to obtain png files
    if mask_folder is None:
        mask_folder = os.path.join(groundtruth_folder, '..', 'mask')

    pred_pngs = scan_png_files(prediction_folder)
    gt_pngs = scan_png_files(groundtruth_folder)
    mask_pngs = scan_png_files(mask_folder)

    pred_diff_gt = set(pred_pngs).difference(gt_pngs)
    assert len(pred_diff_gt) == 0, \
        'No corresponding groundtruth file for the following files:\n' + '\n'.join(pred_diff_gt)
    pred_diff_mask = set(pred_pngs).difference(mask_pngs)
    assert len(pred_diff_mask) == 0, \
        'No corresponding mask file for the following files:\n' + '\n'.join(pred_diff_mask)

    # Measure: mean angle error over all pixels
    mean_angle_error = 0
    total_pixels = 0
    for fname in pred_pngs:
        # print('Proccessing file {}'.format(fname))
        prediction = imageio.imread(os.path.join(prediction_folder, fname))
        groundtruth = imageio.imread(os.path.join(groundtruth_folder, fname))
        mask = imageio.imread(os.path.join(mask_folder, fname)) # Greyscale image
       
        prediction = ((prediction / 255.0) - 0.5) * 2
        groundtruth = ((groundtruth / 255.0) - 0.5) * 2

        total_pixels += np.count_nonzero(mask)
        mask = mask != 0

        a11 = np.sum(prediction * prediction, axis=2)[mask]
        a22 = np.sum(groundtruth * groundtruth, axis=2)[mask]
        a12 = np.sum(prediction * groundtruth, axis=2)[mask]

        cos_dist = a12 / np.sqrt(a11 * a22)
        cos_dist[np.isnan(cos_dist)] = -1
        cos_dist = np.clip(cos_dist, -1, 1)

        angle_error = np.arccos(cos_dist)
        mean_angle_error += np.sum(angle_error)

    return mean_angle_error / total_pixels


def buildModel(x, keep_prob,is_training):
    w1 = weight_variable([3,3,3,512])
    b1 = bias_variable([512])
    conv1 = conv2d(x,w1)+b1
    conv1 = tf.layers.batch_normalization(conv1,training=is_training)
    conv1 = tf.nn.relu(conv1)
    conv2 = tf.nn.dropout(conv1,keep_prob=keep_prob)
    w2 = weight_variable([1,1,512,3])
    b2 = bias_variable([3])
    conv2 = conv2d(conv2,w2)+b2
    conv2 = tf.layers.batch_normalization(conv2,training=is_training)
    output = tf.nn.relu(conv2,name='output')
    return output

data_size = 20000
epochs = 4
data = [i for i in range(data_size)]
batch_size = 20
keep_probability = 0.9

# build the graph
use_batch_norm = True
train_graph = tf.Graph()
with train_graph.as_default():
    x = tf.placeholder(tf.float32,[None, 128,128,3],name='x') # color
    y = tf.placeholder(tf.float32,[None, 128,128,3],name='y') # mask
    z = tf.placeholder(tf.float32,[None, 128,128,3],name='z') # normal labels
    is_training = tf.placeholder(tf.bool,name='is_training') # for batch_normalization
    keep_prob = tf.placeholder(tf.float32,name='keep_prob') # dropout

    output = buildModel(x,keep_prob,is_training)

    loss = 0
    for j in range(batch_size):
        mask = y[j,:,:,:]
        mask_region = tf.not_equal(mask, tf.zeros_like(mask))
        for chn in range(3):
            loss += tf.reduce_mean(tf.boolean_mask(tf.square(output[j,:,:,chn]-z[j,:,:,chn]),mask_region[:,:,chn]))
    # mean_angle_error = 0.0
    # total_pixels = 0

    # for j in range(batch_size):
    #     # nvp = tf.norm(output[j,:,:,:],axis=2)
    #     # nvn = tf.norm(z[j,:,:,:],axis=2)    
    #     # nvp3 = tf.stack([nvp,nvp,nvp],axis=2)
    #     # nvn3  = tf.stack([nvn,nvn,nvn],axis=2)
    #     # print(tf.reduce_max(output[j,:,:,:]))   
    #     # prediction = (tf.divide(output[j,:,:,:],nvp3) - 0.5) * 2.0 ################
    #     # norm = (tf.divide(z[j,:,:,:],nvn3) - 0.5) * 2.0            ################
    #     prediction = (output[j,:,:,:]-0.5)*2
    #     norm = (z[j,:,:,:]-0.5)*2
    #     mask = y[j,:,:,0]
    #     bmask = tf.cast(mask,tf.bool)

    #     total_pixels += tf.count_nonzero(bmask)
        
    #     a11 = tf.boolean_mask(tf.reduce_sum(prediction*prediction, axis=2),bmask)
    #     a22 = tf.boolean_mask(tf.reduce_sum(norm*norm, axis=2),bmask)
    #     a12 = tf.boolean_mask(tf.reduce_sum(prediction*norm, axis=2),bmask)

    #     cos_dist = -1.0*a12 / tf.sqrt(a11 * a22)
    #     # cos_dist = tf.where(tf.is_nan(cos_dist),1.0,cos_dist)
    #     cos_dist = tf.clip_by_value(cos_dist, -1.0, 1.0)
    #     # angle_error = tf.acos(cos_dist)
    #     mean_angle_error += tf.reduce_sum(cos_dist) # -1 the best

    # cost = mean_angle_error / tf.cast(total_pixels,tf.float32)
    # # cost += tf.reduce_mean(tf.boolean_mask(tf.abs(prediction-norm)))
    cost = loss

    if use_batch_norm:
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            opt = tf.train.AdamOptimizer(0.0001).minimize(cost)
    else:
        opt = tf.train.AdamOptimizer(0.0001).minimize(cost)


# the driver
random.shuffle(data)
train, test = train_test_split(data,data_size//20)

train_color = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')
train_mask = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')
train_normal = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')

validation_color = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')
validation_mask = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')
validation_normal = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    min_loss_so_far = 1.57
    for e in range(1,epochs+1):
        num_batches = 0
        los = 0
        every = 5
        for batch_index in get_batches(train,batch_size):
            counter = 0
            for i in batch_index:
                train_color[counter,:,:,:] = readimage('./train/color', i)
                train_mask[counter,:,:,0] = readmask('./train/mask', i)
                train_mask[counter,:,:,1] = readmask('./train/mask', i)
                train_mask[counter,:,:,2] = readmask('./train/mask', i)
                train_normal[counter,:,:,:] = readimage('./train/normal', i)
                
                train_color[counter,:,:,:] = normalize(train_color[counter,:,:,:])
                train_mask[counter,:,:,:] = normalize(train_mask[counter,:,:,:])
                train_normal[counter,:,:,:] = normalize(train_normal[counter,:,:,:])
                # train_color[counter,:,:,:] /= np.amax(train_color[counter,:,:,:]+1)
                # train_mask[counter,:,:,:] /= np.amax(train_mask[counter,:,:,:]+1)
                # train_normal[counter,:,:,:] /= np.amax(train_normal[counter,:,:,:]+1)
                counter += 1

            c, _ = sess.run([cost, opt], feed_dict={x: train_color, y:train_mask, z: train_normal,\
             keep_prob:keep_probability, is_training:True})
            los += c
            num_batches += 1
            if num_batches % every == 0:
                print('Epoch {}/{};'.format(e,epochs),'Batches {}/{};'.format(num_batches,len(train)//batch_size),\
                      'Avg {} batch(es) training loss: {:.3f}'.format(every,los/every))
                los = 0

            if num_batches % 15 == 0:
                vlos = 0
                valid_batches = len(test) // batch_size
                div = 0
                for index in get_batches(test,batch_size):
                    cnt = 0
                    for k in index:
                        validation_color[cnt,:,:,:] = readimage('./train/color', k)
                        validation_mask[cnt,:,:,0] = readmask('./train/mask', k)
                        validation_mask[cnt,:,:,1] = readmask('./train/mask', k)
                        validation_mask[cnt,:,:,2] = readmask('./train/mask', k)
                        validation_normal[cnt,:,:,:] = readimage('./train/normal', k)
                        
                        validation_color[cnt,:,:,:] = normalize(validation_color[cnt,:,:,:])
                        validation_mask[cnt,:,:,:] = normalize(validation_mask[cnt,:,:,:])
                        validation_normal[cnt,:,:,:] = normalize(validation_normal[cnt,:,:,:])
                        # validation_color[cnt,:,:,:] /= (np.amax(validation_color[cnt,:,:,:])+1)
                        # validation_mask[cnt,:,:,:] /= (np.amax(validation_mask[cnt,:,:,:])+1)
                        # validation_normal[cnt,:,:,:] /= (np.amax(validation_normal[cnt,:,:,:]+1)
                        cnt += 1

                    vc,results = sess.run([cost,output], feed_dict={x:validation_color, y:validation_mask, \
                        z: validation_normal, keep_prob:1.0, is_training:False})
                    vlos += vc
                    tmp = 0
                    for k in index:
                        image=Image.fromarray((255.0*results[tmp,:,:,:]).astype(np.uint8))
                        image.save('./train/pred/'+str(k)+'.png')
                        tmp += 1
                    div += 1
                print('Avg validation loss: {:.3f}'.format(vlos/valid_batches))
                valid = evaluate('./train/pred/', './train/normal/', './train/mask/')
                print(valid)
                if valid < min_loss_so_far:
                    min_loss_so_far = valid
                    tf.train.Saver().save(sess, './best_model/surface_normal_est')
                    print('best model saved!\n')