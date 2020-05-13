import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np


def conv(input_tensor, name, kw, kh, n_out, dw=1, dh=1, activation_fn=tf.nn.relu):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        weight = tf.get_variable('weights', [kw,kh,n_in,n_out], tf.float32, xavier_initializer())

        indicesij = np.zeros([kw,kh,n_in,n_out])
        for i in range(3):
            for j in range(3):
                indicesij[i,j,:,:] = 3*(i)+j+1
        '''
        indicesin = np.zeros([1,1,n_in,1])
        indicesout = np.zeros([1,1,1,n_out])
        for i in range(n_in):
            indicesin[:,:,i,:] = i+1
        for i in range(n_out):
            indicesout[:,:,:,i] = i+1
        indices_in = tf.constant(indicesin,dtype=tf.float32)

        indices_out = tf.constant(indicesout,dtype=tf.float32)

        '''
        indices_ij = tf.constant(indicesij,dtype=tf.float32)

        #weights = tf.matmul(indices,tf.matmul(S,tf.linalg.inv(indices),adjoint_b=True))
        weights = tf.multiply(weight[0,0,:,:] , tf.pow(indices_ij,1))+\
                  tf.multiply(weight[0,1,:,:] , tf.pow(indices_ij, 2)) + tf.multiply(weight[0,2,:,:] , tf.pow(indices_ij,3))+ \
                  tf.multiply(weight[1,0,:,:] , tf.pow(indices_ij, 4)) + tf.multiply(weight[1,1,:,:] , tf.pow(indices_ij,5))+ \
                  tf.multiply(weight[1,2,:,:] , tf.pow(indices_ij, 6)) + tf.multiply(weight[2,0,:,:] , tf.pow(indices_ij,7))+ \
                  tf.multiply(weight[2,1:,:] , tf.pow(indices_ij, 8)) + tf.multiply(weight[2,2,:,:] , tf.pow(indices_ij,9))
        #biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        weights = weights/(9**9)
        conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
        activation = activation_fn(conv)
        return activation


def fully_connected(input_tensor, name, n_out, activation_fn=tf.nn.relu):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer())
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        return activation_fn(logits)
def fully_connected_f(input_tensor, name, n_out, activation_fn=tf.nn.relu):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer())
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        return logits




def pool(input_tensor, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='VALID',
                          name=name)


def VGG16(input_layer, rgb_mean, num_class):

    # block 1

    with tf.device('/gpu:0'):
       net = conv(input_layer, name="conv1_1", kh=3, kw=3, n_out=64)
       net = conv(input_layer, name="conv1_2", kh=3, kw=3, n_out=64)
       net = pool(net, name="pool1", kh=2, kw=2, dh=2, dw=2)

   # with tf.device('/gpu:1'):
       # block 2
       net = conv(net, name="conv2_1", kh=3, kw=3, n_out=128)
       net = conv(net, name="conv2_2", kh=3, kw=3, n_out=128)
       net = pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

       # block 3
       net = conv(net, name="conv3_1", kh=3, kw=3, n_out=256)
       net = conv(net, name="conv3_2", kh=3, kw=3, n_out=256)
       net = conv(net, name="conv3_3", kh=3, kw=3, n_out=256)
       net = pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

    #with tf.device('/gpu:3'):
       # block 4
       net = conv(net, name="conv4_1", kh=3, kw=3, n_out=512)
       net = conv(net, name="conv4_2", kh=3, kw=3, n_out=512)
       net = conv(net, name="conv4_3", kh=3, kw=3, n_out=512)
       net = pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

       # block 5
       net = conv(net, name="conv5_1", kh=3, kw=3, n_out=512)
       net = conv(net, name="conv5_2", kh=3, kw=3, n_out=512)
       net = conv(net, name="conv5_3", kh=3, kw=3, n_out=512)
       net = pool(net, name="pool5", kh=2, kw=2, dh=2, dw=2)

       flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
       net = tf.reshape(net, [-1, flattened_shape], name="flatten")

       # FC 1
       net = fully_connected(net, name="fc1", n_out=4096)

       # FC 2
       net = fully_connected(net, name="fc2", n_out=4096)

       # FC 3
       net = fully_connected_f(net, name="fc3", n_out=num_class)
       net = tf.nn.softmax(net)

    return net