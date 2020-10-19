import numpy as np
from PIL import Image
from  config import cfg
import os
from skimage.transform import rescale

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def load_images():
    images_LR = np.zeros([11, 32, 32])
    images_HR = np.zeros([11, 64, 64])
    c = 0
    d = 0
    for filename in np.sort(os.listdir(cfg.DATA_SETS_DIR_1)):
        #print(filename)
        img_LR = Image.open(os.path.join(cfg.DATA_SETS_DIR_1, filename))
        cropped = img_LR.crop((30, 30, 62, 62))
        LR_img = np.array(cropped)
        image_LR = rgb2gray(LR_img)
        #image_LR = rescale(rgb2gray(LR_img),[128/32,128/32])
        #image = image/256
        images_LR[c, :, :] = image_LR
        c +=1
    images_LR = (images_LR)/256
    for filename in np.sort(os.listdir(cfg.DATA_SETS_DIR_2)):
        #print(filename)
        img_HR = Image.open(os.path.join(cfg.DATA_SETS_DIR_2, filename))
        cropped = img_HR.crop((120, 120, 184, 184))
        HR_img = np.array(cropped)
        image_HR = rgb2gray(HR_img)
        # image = image/256
        images_HR[d, :, :] = image_HR
        d += 1
    images_HR = (images_HR)/256
    return np.array(images_LR), np.array(images_HR)
def save_images(set_1, set_2):
    for b in range(len(set_1)):
        im_lr = (Image.fromarray(set_1[b] * 256).convert('L'))
        im_lr.save(cfg.ROOT_DIR + '/outputs1/' + str(b) + '.png')
    for b in range(len(set_2)):
        im_hr = (Image.fromarray(set_2[b] * 256).convert('L'))
        im_hr.save(cfg.ROOT_DIR + '/outputs2/' + str(b) + '.png')
    return im_lr, im_hr
if __name__ == '__main__':
    im_LR, im_HR = load_images()
    #print('Low_resolution')
    print(im_LR.shape)
    #print('High_resolution')
    #print(im_HR)
    save_images(im_LR, im_HR)