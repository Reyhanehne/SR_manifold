import numpy as np
from utils import unpickle
from config import data_directory

meta_data_dict = unpickle(data_directory + "/batches.meta")
cifar_label_names = meta_data_dict[b'label_names']
cifar_label_names_array = np.array(cifar_label_names)
print (cifar_label_names_array)
cifar_test_data_dict = unpickle(data_directory + "/test_batch")
#print (cifar_test_data_dict)
cifar_test_data = cifar_test_data_dict[b'data']
#python3.6 data.py
#print (cifar_test_data)
cifar_test_data = cifar_test_data.reshape(len(cifar_test_data),3,32,32).transpose(0, 2, 3, 1).astype(np.float32)
print (cifar_test_data.shape)
cifar_test_filenames = cifar_test_data_dict[b'filenames']
cifar_test_labels = cifar_test_data_dict[b'labels']
#print (cifar_test_data_dict)
#print (cifar_test_data.shape)
cifar_train_data = None
cifar_train_filenames = []
cifar_train_labels = []
for i in range(1,6):
    cifar_train_data_dict = unpickle(data_directory + "/data_batch_{}".format(i))
    if i == 1:
        cifar_train_data = cifar_train_data_dict[b'data']
    else:
        cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
    #cifar_train_filenames += cifar_train_data_dict[b'filenames']
    cifar_train_labels += cifar_train_data_dict[b'labels']
cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
'''
    cifar_train_data = cifar_train_data_dict[b'data']
    cifar_train_data = cifar_train_data.reshape(len(cifar_train_data), 3, 32, 32)'''

print (cifar_train_data.shape)
'''print (cifar_train_data)
print (len(cifar_train_data_dict))'''

