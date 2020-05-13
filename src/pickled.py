import pickle
def load_cifar10_batch(file):
    with open(file,'rb') as fo:
       data = pickle.load(fo, encoding='bytes')
    return data