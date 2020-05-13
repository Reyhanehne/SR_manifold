import numpy as np
import tensorflow as tf
from model import *
from config import cfg


#Define placeholder for your all inputs to the graph of the network
Input_tensor_xy = tf.compat.v1.placeholder(tf.float32, shape=(None, 32, 32, 1))
#Input_tensor_xyv_HR = tf.compat.v1.placeholder(tf.float32, shape=(None, 32, 32, 3))
Learning_rate = tf.compat.v1.placeholder(tf.float32, shape=None)

coefficients_1 = VGG16_1(Input_tensor_xy, cfg.rgb_mean, cfg.num_out)
#coefficients_2 = VGG16_2(Input_tensor_xy, cfg.rgb_mean, cfg.num_out)
loss_1, f_xy = polyval2d_loss(Input_tensor_xy, coefficients_1, cfg.degree)
#coefficients_2 = VGG16_2(Input_tensor_xy, cfg.rgb_mean, cfg.num_out)
#coefficients = coefficients_1 + coefficients_2
#loss_3, loss_4 = polyval2d_loss(Input_tensor_xyv_HR, coefficients, cfg.degree)
Loss = tf.reduce_mean(loss_1)

#define an optimizer
optimizer = tf.train.GradientDescentOptimizer(Learning_rate)
#optimize your network with the value of the loss function
train_op = optimizer.minimize(Loss)

#Acc = tf.reduce_mean(tf.dtypes.cast(tf.equal(tf.argmax(Logits,axis=1), tf.argmax(Labels,axis=1)),tf.float32))
saver = tf.train.Saver(tf.global_variables())
