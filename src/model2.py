import tensorflow as tf
import numpy as np
from config import cfg
from tensorflow.contrib.layers import xavier_initializer

# def sigmoid(x):
# return 1 / (1 + math.exp(-x))
def fully_connected_f(input_tensor, name, n_out, activation_fn=tf.nn.tanh):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer())
        logits = tf.matmul(input_tensor, weights)
        return logits

def VGG16_1(input_layer, rgb_mean, num_out):
    one = np.zeros([11, cfg.num_out, 1, 1],dtype=np.float32)
    one[:,0,:,:]=1
    input_layer = tf.reshape(input_layer, [-1, 32, 32, 1])
    # input_layer = tf.image.rgb_to_grayscale(input_layer, name=None)
    # define image mean
    # if rgb_mean is None:
    #    rgb_mean = np.array([116.779, 123.68, 103.939], dtype=np.float32)
    # mu = tf.constant(rgb_mean, name="rgb_mean")
    # input_layer = tf.subtract(tf.cast(input_layer, tf.float32), tf.cast(mu, tf.float32), name="input_mean_centered")
    # block 1
    conv1_1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding="same",
                               activation=tf.nn.relu, name='VGG_1_1_1')
    conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_1_2')
    pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

    # block 2
    conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_2_1')
    conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_2_2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

    # block 3
    conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_3_1')
    conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_3_2')
    conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_3_3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2)

    # block 4

    conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_4_1')
    conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_4_2')
    conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_4_3')
    pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2)

    # block 5
    conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_5_1')
    conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_5_2')
    conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               name='VGG_1_5_3')
    pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2)

    # block 6
    flattened_shape = np.prod([s.value for s in pool5.get_shape()[1:]])
    pool5_flat = tf.reshape(pool5, [-1, flattened_shape], name="flatten")
    fc1 = tf.reshape(fully_connected_f(pool5_flat, name='fc',n_out=cfg.num_out),[-1, cfg.num_out, 1, 1])  # *100
    fc1 = fc1 - tf.constant(one)*tf.reshape(fc1[:,0,:,:],[-1,1,1,1])+tf.constant(one)
    '''
    fc2 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc3 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    
    fc4 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc5 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100

    fc6 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc7 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc8 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc9 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc10 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100

    fc11 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc12 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc13 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc14 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc15 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc16 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100

    fc17 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc18 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc19 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc20 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100    fc1 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, 15, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100

    fc21 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc22 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc23 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc24 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc25 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc26 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc27 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc28 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc29 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc30 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc31 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    fc32 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    '''

    # return [fc1,fc2,fc3,fc4,fc5,fc6,fc7,fc8,fc9,fc10,fc11,fc12,fc13,fc14,fc15,fc16,fc17,fc18,fc19,fc20,fc21,fc22,fc23,fc24,fc25,fc26,fc27,fc28,fc29,fc30,fc31,fc32]
    return [fc1]


'''
def VGG16_2(input_layer, rgb_mean, num_out):
    input_layer = tf.reshape(input_layer, [-1, 32, 32, 1])
    #input_layer = tf.image.rgb_to_grayscale(input_layer, name=None)
    # define image mean
    #if rgb_mean is None:
    #    rgb_mean = np.array([116.779, 123.68, 103.939], dtype=np.float32)
    #mu = tf.constant(rgb_mean, name="rgb_mean")
    #input_layer = tf.subtract(tf.cast(input_layer, tf.float32), tf.cast(mu, tf.float32), name="input_mean_centered")
    # block 1
    conv1_1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_1_1')
    conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_1_2')
    pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

    #block 2
    conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_2_1')
    conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_2_2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

    #block 3
    conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_3_1')
    conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_3_2')
    conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_3_3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2)

    #block 4
    conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_4_1')
    conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_4_2')
    conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_4_3')
    pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2)

    #block 5
    conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_5_1')
    conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_5_2')
    conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='VGG_2_5_3')
    pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2)

    #block 6
    flattened_shape = np.prod([s.value for s in pool5.get_shape()[1:]])
    pool5_flat = tf.reshape(pool5, [-1, flattened_shape], name="flatten")
    fc1 = tf.reshape(tf.contrib.layers.fully_connected(pool5_flat, cfg.num_out, activation_fn=None), [-1, cfg.num_out, 1, 1])#*100
    return fc1
'''


def polyval2d_loss(image, weight, degree):
    # image = tf.image.rgb_to_grayscale(image, name=None)
    c = 0
    power_tmp = np.zeros([cfg.num_out, 3]).astype(np.float32)
    for i in range(0, degree+1):
        for j in range(0, degree+1):
            for k in range(0, degree+1):
                if i + j + k <= degree:
                   power_tmp[c, 0] = i
                   power_tmp[c, 1] = j
                   power_tmp[c, 2] = k
                   c = c + 1
    power = tf.constant(power_tmp)

    a = np.zeros([32, 32, 2]).astype(np.float32)
    for h in range(32):
        a[h, :, 0] = (-.5 + h / 32) * 2
        a[:, h, 1] = (-.5 + h / 32) * 2
    A = tf.constant(a)

    rand_tmp = np.arange(256,dtype=np.float32)
    rand_l = tf.random.shuffle(tf.constant(rand_tmp))
    rand_u = tf.random.shuffle(tf.constant(rand_tmp))
    l_xy = 0.0
    u_xy = 0.0
    img_l = image[:, :, :, 0] - rand_l[0]/256
    img_u = image[:, :, :, 0] + rand_u[0]/256
    img_l = tf.nn.relu(img_l)
    img_u = -tf.nn.relu(-(img_u-1))+1

    for w in range(1):
        f_xy = 0.0
        for d in range(cfg.num_out):
            f_xy += (weight[w][:, d, :, :] * (A[:, :, 0] ** power[d, 0]) * (A[:, :, 1] ** power[d, 1])) * (
                    image[:, :, :, 0] ** power[d, 2])
            l_xy += (weight[w][:, d, :, :] * (A[:, :, 0] ** power[d, 0]) * (A[:, :, 1] ** power[d, 1])) * (
                    (img_l) ** power[d, 2])
            u_xy += (weight[w][:, d, :, :] * (A[:, :, 0] ** power[d, 0]) * (A[:, :, 1] ** power[d, 1])) * (
                    (img_u) ** power[d, 2])

    Loss = tf.square(f_xy) + (tf.square(tf.tanh(l_xy)+1)+tf.square(tf.tanh(u_xy)-1))/2


    return Loss, f_xy
'''
def finding_root_poly(image, weight, degree)
    c = 0
    power_tmp = np.zeros([cfg.num_out, 3]).astype(np.float32)
    for i in range(0, degree + 1):
        for j in range(0, degree + 1):
            for k in range(0, degree + 1):
                if i + j + k <= degree:
                    power_tmp[c, 0] = i
                    power_tmp[c, 1] = j
                    power_tmp[c, 2] = k
                    c = c + 1
    power = tf.constant(power_tmp)
    a = np.zeros([32, 32, 2]).astype(np.float32)
    for h in range(32):
        a[h, :, 0] = (-.5 + h / 32) * 2
        a[:, h, 1] = (-.5 + h / 32) * 2
    A = tf.constant(a)
    for d in range(cfg.num_out):
        coeff
      np.roots(weight[:, d, :, :] * (A[:, :, 0] ** power[d, 0]) * (A[:, :, 1] ** power[d, 1]))
'''
