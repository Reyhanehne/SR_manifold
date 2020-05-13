import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_cfar10_batch(file):
    """load the cifar-10 data"""
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data