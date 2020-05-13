import numpy as np
from pickled import load_cifar10_batch
from  config import cfg
#import tensorflow as tf
#from skimage import color
from skimage.transform import rescale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def data_train():
    cifar_train_data_dict = load_cifar10_batch(cfg.DATA_SETS_DIR + "data_batch_{}".format(1))
    images = cifar_train_data_dict[b'data'].reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
    #cifar_train_labels = cifar_train_data_dict[b'labels']
    #labels = np.eye(cfg.num_class)[np.array(cifar_train_labels)]
    for i in range(2,6):
        cifar_train_data_dict = load_cifar10_batch(cfg.DATA_SETS_DIR + "data_batch_{}".format(i))
        image = cifar_train_data_dict[b'data'].reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
        images = np.vstack([images,image])
        '''cifar_train_labels = cifar_train_data_dict[b'labels']
        label = np.eye(cfg.num_class)[np.array(cifar_train_labels)]
        labels = np.vstack([labels,label])'''
        images_LR = np.zeros([images.shape[0], 32, 32])
        for n in range(images.shape[0]):
            images_LR[n] = rescale(rgb2gray(images[n]),[1/1,1/1])
        #images_LR = rgb2gray(images[n])
            #images_LR[n] = images_LR[n]/255
        images_LR = np.array(images_LR) / 256

    #images = np.array(images) / 255
    #images_HR = rgb2gray(images)
    #images_LR = images_HR.resize((8, 8))
   #images_LR = np.resize(images_HR,(8,8))
        return images_LR #, images_HR

def data_test():
    cifar_test_data_dict = load_cifar10_batch(cfg.DATA_SETS_DIR + "test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    #cifar_test_labels = cifar_test_data_dict[b'labels']
    #labels = np.eye(cfg.num_class)[np.array(cifar_test_labels)]
    images = cifar_test_data.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
    images = np.array(images) / 256
    images_LR = rgb2gray(images)
    #images_LR = images_HR.resize(
    #   (32, 32))
    #images_LR = np.resize(images_HR, (8, 8))
    return images_LR #, images_HR






