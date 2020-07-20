import numpy as np
from PIL import Image
from  config import cfg
import os
#from scipy import misc
#from skimage.transform import rescale

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def data_train_1():
    images = []
    for filename in os.listdir(cfg.DATA_SETS_DIR):
        print(filename)
        img = Image.open(os.path.join(cfg.DATA_SETS_DIR, filename))
        #for i in range(batch_size):
        #img = Image.open(cfg.DATA_SETS_DIR + str(i).zfill(2) + '.jpg')
        lr_img = np.array(img)

        image = rgb2gray(lr_img)/256
        images.append(image)
        #img = img.save(cfg.DATA_SETS_DIR + str(i).zfill(2) + '.jpg')
    images = np.array(images).reshape(-1,32,32)

    return images