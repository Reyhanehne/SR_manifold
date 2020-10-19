import numpy as np
from config import cfg
from bicubic_pytorch import core
import torch



def poly_loss(image, weight, degree):
    image_bd = core.imresize(image, scale=0.5)
    image_up_bd = core.imresize(image_bd, scale=2)
    c = 0
    power_tmp = np.zeros([cfg.num_out,2]).astype(np.float32)
    for i in range(0, degree +1):
        for j in range(0, degree + 1):
            if i+j <= degree:
                power_tmp[c,0]=i
                power_tmp[c,1]=j
                c = c+1
    power = torch.IntTensor(power_tmp)

    a = np.zeros([32, 32, 2]).astype(np.float32)
    for i in  range(32):
        a[i, :, 0] =(-.5+ i /32)*2
        a[:, i, 1] =(-.5+ i /32)*2
    A = torch.IntTensor(a)
    f_xy_list = []
    f_xy = 0.0
    for k in range(cfg.num_out):
        f_xy += weight[:, k, :, :] * (A[:, :, 0] ** power[k, 0]) * (A[:, :, 1] ** power[k, 1])
        #f_xy = f_xy / (tf.reshape(tf.reduce_mean(tf.reshape(f_xy, [-1, 32 * 32]), 1) + 0.00001, [-1, 1, 1]))
    f_xy_list.append(f_xy)
    image_lr = image_up_bd[:,:,:,0] + f_xy_list
    loss_1 = torch.square(image[:,:,:,0] -  image_lr[:,:,:,0])

    return loss_1, image_lr
