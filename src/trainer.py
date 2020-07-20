import numpy as np
import argparse
import tensorflow as tf
from config import *
from train import *
from data import *
from sklearn.utils import shuffle
import os
from PIL import Image

def real_finder(r):
    Imag = r.imag
    Real = r.real
    mask = np.where(Imag==0)
    Real_out = Real[mask]
    return Real_out

c = 0
power = np.zeros([cfg.num_out, 3]).astype(np.float32)
for i in range(0, cfg.degree + 1):
    for j in range(0, cfg.degree + 1):
        for k in range(0, cfg.degree + 1):
            if i + j + k <= cfg.degree:
                power[c, 0] = i
                power[c, 1] = j
                power[c, 2] = k
                c = c + 1

A = np.zeros([32, 32, 2]).astype(np.float32)
for h in range(32):
    A[h, :, 0] = (-.5 + h / 32) * 2
    A[:, h, 1] = (-.5 + h / 32) * 2

class Trainer():
    def __init__(self,dir_name, learning_rate, batch_size, total_epochs=400, weights=None):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if weights:
                ckpt = tf.train.get_checkpoint_state(os.path.join(cfg.CHECKPOINT_DIR, tag))
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(cfg.LOG_DIR, tag), sess.graph)
            #Set GPU parameters
            #train_data_LR, train_data_HR = data_train()
            #train_data_LR = data_train()
            #train_data_LR = data_train_1(batch_size)
            train_data_LR = data_train_1()
            for e in range(total_epochs):
                loss_epoch = 0
                #acc_epoch = 0
                #if e ==100:
                    #learning_rate = learning_rate/10
                #train_data_LR, train_data_HR = shuffle(train_data_LR, train_data_HR, random_state=e)
                #train_data_LR= shuffle(train_data_LR, random_state=e)
                step = int(len(train_data_LR)/batch_size)
                #for s in range(int(step)):
                for s in range(step):
                    batch_img = train_data_LR[s*batch_size:(s+1)*batch_size]

                    #load data
                    #batch_images_LR, batch_images_HR = train_data_LR[s*batch_size:(s+1)*batch_size], train_data_HR[s*batch_size:(s+1)*batch_size]
                    #batch_images_LR = train_data_LR[s*batch_size:(s+1)*batch_size]
                    #print((batch_images_LR.shape))

                    '''   B = np.zeros([batch_images_HR.shape[0], batch_images_HR.shape[1], batch_images_HR.shape[2], 3])
                                      B[:, :, :, 0] = np.arange(32) / 32
                                      B[:, :, :, 1] = np.arange(32) / 32
                                      B[:, :, :, 2] = batch_images_HR   '''
                    feed_dict = {Input_tensor_xy: np.reshape(batch_img,[-1,32,32,1]),
                                 Learning_rate: learning_rate
                                 }
                                 #,Input_tensor_xyv_HR: B}
                    _, loss_batch, coeff = sess.run([train_op, Loss,  coefficients_1], feed_dict=feed_dict)                    #print('epoch:    ', e, '   step:  ', s, '   acc_batch:   ',acc_batch, '   loss_batch:   ',loss_batch)
                    loss_epoch += loss_batch
                    print(coeff)
                    #print(f_xy_np[0])
                    #acc_epoch += acc_batch

                    Coeff_new = np.zeros([batch_size,32,32,cfg.degree+1])

                    for c in range(cfg.num_out):
                        d = int(power[c,2])
                        Coeff_new[:,:,:,d] += coeff[0][:, c, :, :] * (A[:, :, 0] ** power[c, 0]) * (A[:, :, 1] ** power[c, 1])
                    image_pred = np.zeros([batch_size,32,32])
                    image_qred = np.zeros([batch_size,32,32])
                    image_ored = np.zeros([batch_size,32,32])
                    for b in range(batch_size):
                        for h in range(32):
                            for w in range(32):
                                r = np.roots(Coeff_new[b,h,w,:])
                                r_real = real_finder(r)
                                r_real = r_real[r_real > 0]
                                r_real = r_real[r_real <= 1]
                                r_real = r_real*256
                                r_real = r_real.astype(np.uint8)
                                r_real = np.setdiff1d(r_real, 0)
                                if len(r_real)>0:
                                    p = np.mean(r_real)
                                    q = np.min(r_real)
                                    o = np.max(r_real)
                                if len(r_real)==0:
                                    p = 0
                                    q = 0
                                    o = 0
                                image_pred[b,h,w] = p
                                image_qred[b,h,w] = q
                                image_ored[b,h,w] = o

                    for b in range(batch_size):
                        im_p = (Image.fromarray(image_pred[b]).convert('L'))
                        im_p.save(cfg.ROOT_DIR+'/outputs1/'+str(b)+'.jpeg')
                        im_q = (Image.fromarray(image_qred[b]).convert('L'))
                        im_q.save(cfg.ROOT_DIR+'/outputs2/'+str(b)+'.jpeg')
                        im_o = (Image.fromarray(image_ored[b]).convert('L'))
                        im_o.save(cfg.ROOT_DIR+'/outputs3/'+str(b)+'.jpeg')
                    #im_GT = (Image.fromarray(np.reshape(batch_images_LR[0]*256,[32,32]).astype(np.uint8)).convert('L'))
                    #im_GT.save(cfg.DATA_SETS_DIR+'outputs2/_'+str(s)+'_GT.jpeg')

                #acc_epoch = acc_epoch/step
                #loss_epoch = loss_epoch/step

                print('\nEpoch {} Loss: {}'.format(e + 1, loss_epoch))
                train_summary = tf.Summary(
                    value=[tf.Summary.Value(tag='train_loss', simple_value=loss_epoch)]
                )

                summary_writer.add_summary(summary=train_summary, global_step=e)
                summary_writer.flush()
                #Compute the average of the loss and accuracy among all the data in an epoch
                #Save the parameters to the cpkt
                #give the acc and loss to the tensorbaord
                saver.save(sess=sess, save_path=os.path.join(cfg.CHECKPOINT_DIR, tag, 'Net.ckpt'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training VGG-16 on cifar10')

    parser.add_argument('-w', '--weights', type=str, nargs='?', default=True,
                        help='use pre trained weights example: -w True')

    parser.add_argument('-ep', '--max_epoch', type=int, nargs='?', default=10000,
                        help='max count of train epoch')

    parser.add_argument('-lr', '--init_learning_rate', type=int, nargs='?', default=0.01,
                        help='learning rate exampe: -lr 0.01')

    parser.add_argument('-bs', '--batch_size', type=int, nargs='?', default=11,
                        help='batch size exampe: -bs 64')

    parser.add_argument('-dir', '--direct_name', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')


    args = parser.parse_args()

    max_epoch = args.max_epoch
    init_learning_rate = args.init_learning_rate
    batch_size = args.batch_size

    print('\n\n{}\n\n'.format(args))
    direct_name = args.direct_name
    if direct_name == 'unknown_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' % tag)

    max_epoch = args.max_epoch
    #weights = args.weights
    weights = []

    Trainer(dir_name=tag, learning_rate=init_learning_rate, batch_size=batch_size, total_epochs=max_epoch, weights=weights)