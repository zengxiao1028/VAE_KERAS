import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

import Pill_Generator
np.random.seed(0)
tf.set_random_seed(0)

input_dim = 3
batch_size = 128
latent_dim = 2



def create_network():
    #input
    x = tf.placeholder(tf.float32, [None, 100,100,3])

    with slim.arg_scope([slim.convolution2d]):
        #encoder
        conv1 = slim.repeat(x, 2, slim.conv2d, 32, 3, scope='conv1')
        pool1 = slim.max_pool2d(conv1, 3)

        conv2 = slim.repeat(pool1,2,slim.conv2d,64,3,scope='conv2',padding='VALID')
        pool2 = slim.max_pool2d(conv2,3)

        conv3 = slim.repeat(pool2,2,slim.conv2d,32,3,scope='conv3',padding='VALID')
        pool3 = slim.max_pool2d(conv3, 3)
        conv3_flat = slim.flatten(pool3)

        encoder_fc = slim.fully_connected(conv3_flat, 128)

        encoder_mu = slim.fully_connected(encoder_fc, latent_dim, scope='encoder_mu', activation_fn=None)
        encoder_sigma_square = tf.square(slim.fully_connected(encoder_fc, latent_dim, scope='encoder_sigma',
                                                              activation_fn=None))
        encoder_sigma = tf.sqrt(encoder_sigma_square)

        #sample z
        eps = tf.random_normal((batch_size, latent_dim), 0, 1,dtype=tf.float32)
        z = tf.add(encoder_mu,tf.mul(encoder_sigma, eps))

        #decoder
        decoder_fc = slim.fully_connected(z, 128)
        decoder_upsample = slim.fully_connected(decoder_fc, 8*8*32)
        decoder_upsample = tf.reshape(decoder_upsample,[-1,8,8,32])

        deconv1 = slim.repeat(decoder_upsample, 2, slim.conv2d_transpose, 64, 3, scope='deconv1', padding='SAME')
        deconv1_unpool = tf.image.resize_images(deconv1,conv3.get_shape().as_list()[1:3])

        deconv2 = slim.repeat(deconv1_unpool, 2, slim.conv2d_transpose, 32, 3, scope='deconv2', padding='SAME')
        deconv2_unpool = tf.image.resize_images(deconv2, conv2.get_shape().as_list()[1:3])

        deconv3 = slim.repeat(deconv2_unpool,2, slim.conv2d_transpose, 32, 3, scope='deconv3', padding='SAME')
        deconv2_unpool = tf.image.resize_images(deconv2, conv2.get_shape().as_list()[1:3])

        decoder_rec_x = slim.fully_connected(decoder_fc, input_dim, scope='dncoder_rec_x',activation_fn=None)

    #loss
    recon_loss = tf.reduce_sum(tf.squared_difference(x, decoder_rec_x),axis=[1])
    latent_loss = tf.reduce_sum(0.5 * (tf.square(encoder_mu) + encoder_sigma_square -  tf.log(encoder_sigma_square + 1e-8 ) - 1.0),axis=[1])
    loss = tf.reduce_mean(recon_loss + latent_loss)
    return {'x':x,  'loss':loss, 'rec_loss':recon_loss, 'latent_loss':latent_loss,'rec_x':decoder_rec_x}



def train():

    ops = create_network()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(ops['loss'])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    fig = plt.figure(figsize=(10,4))
    plt.ion()


    xs = []
    ys = []

    X_train, X_test = Pill_Generator.get_batch(batch_size=batch_size)
    for i in range(20000):

        X_train_images = sess.run(X_train)
        _, cost, rec_loss, latent_loss = sess.run([optimizer, ops['loss'],ops['rec_loss'], ops['latent_loss']], feed_dict={ops['x']: X_train_images})

        if i % 20 == 0 :
            xs.append(i)
            ys.append(cost)

        if i % 500 == 0:
            X_test_images = sess.run(X_test)
            cost, rec_x, rec_loss, latent_loss = sess.run([ops['loss'], ops['rec_x'], ops['rec_loss'], ops['latent_loss']],
                feed_dict={ops['x']: X_test_images})
            print ( 'step: %04d' %  (i), "cost=", "{:.4f}".format(cost))
            print('Rec loss: %.4f   Latent loss: %.4f' % (np.mean(rec_loss),np.mean(latent_loss)))
            plt.subplot(121)
            plt.imshow(X_test_images[0])
            plt.subplot(122)
            plt.imshow(rec_x[0])
            plt.pause(0.001)

    fig.savefig('pill.pdf', bbox_inches='tight')
    return xs, ys


if __name__=='__main__':

    xs,ys = train()

    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111)
    ax.plot(xs,ys)
    ax.set_title('Variational Free Engergy')
    fig_2.savefig('VFE.pdf', bbox_inches='tight')
