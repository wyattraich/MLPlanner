from __future__ import print_function, division
import scipy
import scipy.misc
import imageio

from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader2 import DataLoader
import numpy as np
import os
import tensorflow as tf

#notes:
# cannot use for loops or anything that is tf. or K. because tensorflow needs to be
# able to run backprop. There is a wrapper function tf.map_fn but can only run
# on cpu, not GPU and generally has issues with complex functions. Is best to avoid
# implemented matrix op function to determine continuity. fails when the path cannot
# represented as either f(x) = x or f(y)=y ie is not one-to-one with regard to
# x or y axis. Only 13 out of 800 training examples

def pixel_wise(y_true,y_pred):

    #A = tf.constant([[0.1,0.0,0.0,0],
    #                 [1.0,0.0,0.0,0],
    #                 [1.0,0.0,0.0,1],
    #                 [1.0,1.0,1.0,1]]
    #                 ,dtype=tf.float64)

    A = y_pred[2:254,2:254,1] #get green layer of pred image
    #At = np.transpose(A) #transpose for second ck

    #plt.figure()
    #plt.imshow(A)
    #plt.figure()
    #plt.imshow(At)
    #plt.show()

    sess = tf.Session() #initialize tf session
    with sess.as_default():

        zero = tf.constant(0, dtype=tf.float64) # init constant 0, float64
        p6 = tf.constant(0.5, dtype=tf.float64)
        where = tf.math.greater(A, p6) #get bool tensor of every place that has any green
        indices = tf.where(where) #get indicies of locations with green
        shift = tf.roll(indices,shift=1,axis=0) #shift indices down one
        diff_pre = tf.subtract(indices,shift) #subtract indicies from shifted

        #get rid of first row in diff as this row causes false discontinuities after shift
        one = tf.constant(1, dtype=tf.int64) #init constant 1, int64
        one_f = tf.constant(1, dtype=tf.float64)
        two = tf.constant(2,dtype=tf.int32) #init constant 2, int64
        siz = tf.size(diff_pre) #get length of diff matrix (both cols)
        len = tf.divide(siz,two)
        len_m_one = tf.math.subtract(len,one_f) #subract one from num rows for padding 1 vec

        is_empty = tf.equal(tf.size(indices), 0)
        len_m_one = tf.cond(is_empty, lambda: tf.constant(0,dtype=tf.float64), lambda: len_m_one)

        one_v1 = tf.ones([1,2],dtype=tf.int64) #init [1,1]
        one_v2 = tf.ones([len,2],dtype=tf.int64)# init [n,2] ones to subtract from vec
        inv_mult_mat = tf.pad(one_v1, [[0, len_m_one], [0, 0]]) #apply paddings to one_v1
        mult_mat = tf.abs(tf.subtract(inv_mult_mat,one_v2)) #create desired vec to get rid of first row in diff
        diff_post1 = tf.multiply(mult_mat,diff_pre) #multiply mat by diff to remove first row

        where_one = tf.math.greater(diff_post1, one) # find where there are discontinuities (bool tensor)
        discont_ind = tf.where(where_one) #get indicies of where discont
        #need to add line to get actual diff matrix values of discontinuities
        discont_1 = K.sum(discont_ind)

        A = tf.transpose(A)#transpose A to check rot90 of above
        where = tf.math.greater(A, p6) #get bool tensor of every place that has any green
        indices = tf.where(where) #get indicies of locations with green
        shift = tf.roll(indices,shift=1,axis=0) #shift indices down one
        diff_pre2 = tf.subtract(indices,shift) #subtract indicies from shifted

        #get rid of first row
        diff_post2 = tf.multiply(mult_mat,diff_pre2)

        where_one = tf.math.greater(diff_post2, one) # find where there are discontinuities (bool tensor)
        discont_ind = tf.where(where_one) #get indicies of where discont
        discont_2 = K.sum(discont_ind)
        #discont_2 = K.sum(discont_val)  #sum them all up

        discont_1_g = tf.math.less(discont_1,one) #true if no discont found
        discont_2_g = tf.math.less(discont_2,one) #false if discont found

        d = tf.math.logical_or(discont_1_g, discont_2_g) #if either one or both is 1: no discont, if both are false: discont
        # return the sum of both discontinuites if it is discontinuous. if it is continuous, return zero loss
        result = tf.cond(d, lambda: zero, lambda: tf.math.add(tf.cast(discont_1, tf.float64),tf.cast(discont_2, tf.float64)))
        result = tf.cond(is_empty, lambda: tf.constant(1000, dtype=tf.float64), lambda: result)

        result = result.eval() #eval result so it can be returned

        #two = tf.constant(2, dtype=tf.int32)
        #zeros = tf.constant([0,0], dtype=tf.int32)

        #print(type(len_m_one.eval()))
        #print(indices.eval())
        #print(is_empty.eval())
        #print(len_m_one.eval())
        #print(siz.eval())
        #print(len.eval())
        #print(mult_mat.eval())
        #print(diff_pre.eval())
        #print(diff_post1.eval())
        #print(indices.eval())
        #print(shift.eval())
        #print(diff.eval())
        #print(d.eval())
        #print(discont_1.eval())
        #print(discont_2.eval())
        #print(discont_1_g.eval())
        #print(discont_2_g.eval())
        #print(d.eval())
        #print(result.eval())

    return result

def fn_true(a):
    a = tf.constant(0,dtype=tf.float64)

if __name__ == '__main__':
    img_test = imageio.imread('line_1.png', pilmode='RGB').astype(np.float)
    imgs_test = []

    imgs_test.append(img_test)
    imgs_test = np.array(imgs_test)/127.5 - 1.

    loss = []

    for i in range(215,216):
        #a = '//Users/wyattraich/Desktop/work/mike_fast_march/path_line/p_%d.png'% i
        a = './line_1_d.png'
        #a = './6.png'
        img_test = imageio.imread(a, pilmode='RGB').astype(np.float)

        plt.figure()
        plt.imshow(img_test)

        plt.figure()
        plt.imshow(img_test[:,:,1])
        plt.show()

        loss.append(pixel_wise(img_test,img_test))

    #find non zeros
    print(loss)
    wrong_ind = np.nonzero(loss)
    #print(wrong_ind)
