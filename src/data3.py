import numpy as np
from PIL import Image
from  config import cfg
from skimage.transform import rescale

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def data_train_2():
    img = Image.open(cfg.DATA_SETS_DIR+'/DIV2K_train_HR/0251.png')
    cropped = img.crop((700, 700, 1600, 1600))
    #cropped.save(cfg.DATA_SETS_DIR+'/DIV2K_train_HR/060400.png')
    lr_img = np.array(cropped)
    image = rescale(rgb2gray(lr_img ),(32/900,32/900))
    image = image/ 256
    return image
