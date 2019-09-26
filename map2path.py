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

#def pixel_wise(y_true,y_pred):

    #A = y_pred[0,:,:,1]
    #zero = tf.constant(0, dtype=tf.float32)
    #where = tf.not_equal(A, zero)
    #indices = tf.where(where)

    #tf.keras.backend.print_tensor(indices)

    #return 0

    #return 10*K.sum(K.abs(y_true[0,:,:,1] - y_pred[0,:,:,1])) + K.sum(K.abs(y_true - y_pred))
    #add regularization connectivity test, connect adjacent green pixels in tree, ck for cont
    #metric based on ground truth line

def pixel_wise(y_true,y_pred):

    A = y_pred[0,:,:,1]

    #zero = tf.constant(0, dtype=tf.float32) # init constant 0, float64
    zero = tf.Variable(0, dtype=tf.float32)
    zero.assign(0)
    zero_64 = tf.Variable(0, dtype=tf.float64)
    zero.assign(0)
    #p6 = tf.constant(0.5, dtype=tf.float32)
    p6 = tf.Variable(0.5, dtype=tf.float32)
    p6.assign(0.5)
    where = tf.math.greater(A, p6) #get bool tensor of every place that has any green
    indices = tf.where(where) #get indicies of locations with green
    shift = tf.roll(indices,shift=1,axis=0) #shift indices down one
    diff_pre = tf.subtract(indices,shift) #subtract indicies from shifted

    #get rid of first row in diff as this row causes false discontinuities after shift
    #one = tf.constant(1, dtype=tf.int64) #init constant 1, int64
    one = tf.Variable(1, dtype=tf.int64)
    one.assign(1)
    #one_f = tf.constant(1, dtype=tf.float64)
    one_f = tf.Variable(1, dtype=tf.float64)
    one_f.assign(1)
    #two = tf.constant(2,dtype=tf.int32) #init constant 2, int64
    two = tf.Variable(2,dtype=tf.int32)
    two.assign(2)
    siz = tf.size(diff_pre) #get length of diff matrix (both cols)
    len = tf.divide(siz,two)
    len_m_one = tf.math.subtract(len,one_f) #subract one from num rows for padding 1 vec

    is_empty = tf.equal(tf.size(indices), 0)
    #len_m_one = tf.cond(is_empty, lambda: tf.constant(0,dtype=tf.float64), lambda: len_m_one)
    len_m_one = tf.cond(is_empty, lambda: zero_64, lambda: len_m_one)

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
    result = tf.cond(d, lambda: zero, lambda: tf.math.add(tf.cast(discont_1, tf.float32),tf.cast(discont_2, tf.float32)))
    #result = tf.cond(is_empty, lambda: tf.constant(10000, dtype=tf.float32), lambda: result)
    one_thou = tf.Variable(10000, dtype=tf.float32)
    one_thou.assign(1000)
    result = tf.cond(is_empty, lambda: one_thou, lambda: result)

    #L1 loss addition
    abs_sum = K.sum(K.abs(y_true[0,:,:,1] - y_pred[0,:,:,1]))
    abs_sum_32 = tf.cast(abs_sum, dtype=tf.float32)
    #scale_l1 = tf.constant(0.01, dtype=tf.float32)
    scale_l1 = tf.Variable(0.01, dtype=tf.float32)
    scale_l1.assign(0.01)
    #scale_discont = tf.constant(0.0001, dtype=tf.float32)
    scale_discont = tf.Variable(0.0001, dtype=tf.float32)
    scale_discont.assign(0.0001)
    L1 = tf.multiply(scale_l1,abs_sum_32)
    cust = tf.multiply(scale_discont,result)

    return tf.add(cust,L1)


def pixel_wise_np(y_true,y_pred):
    return 0.01*np.sum(np.absolute(np.subtract(y_true[0,:,:,1], y_pred[0,:,:,1]))) + 0.001*np.sum(np.absolute(np.subtract(y_true, y_pred)))

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'paths'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', pixel_wise],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
        #self.combined.compile(loss=['mse', 'mae'],

        #self.generator.compile(loss=pixel_wise, loss_weights=[100], optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        accuracy = []
        epoch_vec = []
        G_loss = []
        D_loss = []
        G_loss_cust = []

        for epoch in range(epochs):

            accuracy_prev = accuracy

            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                #print(imgs_A.shape)
                #plt.figure(0)
                #plt.imshow(imgs_A[0,:,:,1])
                #plt.show()

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                #plt.figure(0)
                #plt.imshow(imgs_A[0,:,:,1])
                #plt.imshow(imgs_A[0,:,:,1])
                #plt.figure(1)
                #plt.imshow(fake_A[0,:,:,1])
                #plt.imshow(fake_A[0,:,:,1])
                #plt.show()

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators ****?????
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                #plt.figure()
                #plt.imshow(fake_A[0,:,:,1])
                #plt.show()

                asdf = pixel_wise(imgs_A,fake_A)

                #sess = tf.Session() #initialize tf session
                #with sess.as_default():
                    #print(asdf.eval())
                #g_loss_np = pixel_wise_np(imgs_A,fake_A)

                #apply custom loss on generator
                #gen_loss = self.generator.train_on_batch([imgs_A],[fake_A])

                #print(imgs_A[0,:,:,1])
                #print(np.sum(np.abs(imgs_A[0,:,:,1] - fake_A[0,:,:,1])))
                #print(K.sum(K.abs(imgs_A[0,:,:,1] - fake_A[0,:,:,1])))

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] [G loss_cust: %f] [Gen_np loss: ] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0], g_loss[2], elapsed_time))

                accuracy.append(100*d_loss[1])
                epoch_vec.append(epoch)
                G_loss.append(g_loss[0])
                D_loss.append(d_loss[0])
                G_loss_cust.append(g_loss[2])


                #if d_loss[1] > accuracy:
                 #   accuracy = d_loss[1]

                # If at save interval => save generated image samples
                #if epoch % sample_interval == 0 and batch_i == 1:
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

            #if accuracy >= accuracy_prev:
             #   if os.path.exists("saved_model/gen_model%d_line.h5" % (epoch-1)):
              #      os.remove("saved_model/gen_model%d_line.h5" % (epoch-1))
               #     os.remove("saved_model/both_model%d_line.h5" % (epoch-1))
                #    os.remove("saved_model/dis_model%d_line.h5" % (epoch-1))


        self.generator.save("saved_model/gen_model%d_line_cust.h5" % (epoch))
        self.combined.save("saved_model/both_model%d_line_cust.h5" % (epoch))
        self.discriminator.save("saved_model/dis_model%d_line_cust.h5" % (epoch))

        plt.figure(1)
        plt.plot(epoch_vec,accuracy)
        plt.title('Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig('saved_plots/accuracy.png')

        plt.figure(2)
        plt.plot(epoch_vec,G_loss)
        plt.title('Generator Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Generator Loss')
        plt.savefig('saved_plots/g_loss.png')

        plt.figure(3)
        plt.plot(epoch_vec,D_loss)
        plt.title('Discriminator Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Discriminator Loss')
        plt.savefig('saved_plots/d_loss.png')

        plt.figure(4)
        plt.plot(epoch_vec,G_loss_cust)
        plt.title('Generator custom Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Generator custom Loss')
        plt.savefig('saved_plots/g_loss_cust.png')

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d_line_n_cont.png" % (self.dataset_name, epoch, batch_i))
        plt.close()



if __name__ == '__main__':
    #train
    gan = Pix2Pix()
    gan.train(epochs=20, batch_size=1, sample_interval=200)

    """
    model = load_model("saved_model/gen_model_line.h5")

    #a = DataLoader("paths",img_res=(256, 256))

    #imgs_test, img_true = a.load_data(batch_size=1, is_testing=True)

    #print(img_test)

    img_test = imageio.imread('4.png', pilmode='RGB').astype(np.float)
    imgs_test = []

    imgs_test.append(img_test)
    imgs_test = np.array(imgs_test)/127.5 - 1.

    fake_A = model.predict(imgs_test)

    plt.figure(1)
    plt.imshow(imgs_test[0])
    #plt.figure(2)
    #plt.imshow(img_true[0])
    plt.figure(2)
    plt.imshow(fake_A[0])
    plt.show()

    """
