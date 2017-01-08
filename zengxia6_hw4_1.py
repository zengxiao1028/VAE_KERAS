import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import MyConfig
import Pill_Generator
import os
import logging
np.random.seed(0)
tf.set_random_seed(0)


batch_size = 128
latent_dim = 32



def create_network():
    #input
    x = tf.placeholder(tf.float32, [None, 100,100,3])

    with slim.arg_scope([slim.convolution2d,slim.fully_connected,
                         slim.convolution2d_transpose],weights_regularizer=slim.l2_regularizer(0.0005)):
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

        deconv1_upsample = tf.image.resize_images(decoder_upsample, conv3.get_shape().as_list()[1:3])
        deconv1 = slim.repeat(deconv1_upsample, 2, slim.conv2d_transpose, 64, 3, scope='deconv1')

        deconv2_unpool = tf.image.resize_images(deconv1, conv2.get_shape().as_list()[1:3])
        deconv2 = slim.repeat(deconv2_unpool, 2, slim.conv2d_transpose, 32, 3, scope='deconv2')

        deconv3_unpool = tf.image.resize_images(deconv2, conv1.get_shape().as_list()[1:3])
        deconv3 = slim.convolution2d_transpose(deconv3_unpool,num_outputs=32, kernel_size=3 )
        deconv3 = slim.convolution2d_transpose(deconv3,num_outputs=3, kernel_size=3,activation_fn=None)
        rec_x = tf.nn.sigmoid(deconv3)

    #loss
    loss = tf.nn.sigmoid_cross_entropy_with_logits(deconv3, x, name=None)
    #loss = tf.squared_difference(rec_x,x)
    recon_loss = tf.reduce_mean(tf.reduce_sum(loss,axis=[1,2,3]))

    regularization_loss = tf.add_n(slim.losses.get_regularization_losses())

    latent_loss = tf.reduce_mean(tf.reduce_sum(0.5 * (tf.square(encoder_mu) + encoder_sigma_square -  tf.log(encoder_sigma_square + 1e-8 ) - 1.0),axis=[1]))

    loss = recon_loss + latent_loss + regularization_loss

    return {'x':x,  'loss':loss, 'rec_loss':recon_loss, 'latent_loss':latent_loss,'rec_x':rec_x , 'reg_loss':regularization_loss}



def train():

    ops = create_network()
    X_train = Pill_Generator.get_batch(MyConfig.pill_images_path, batch_size=batch_size)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(ops['loss'])

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    fig = plt.figure(figsize=(10, 4))
    plt.ion()

    i = 0
    try:
        while not coord.should_stop():

                X_train_images = sess.run(X_train)

                _ = sess.run([optimizer],feed_dict={ops['x']: X_train_images})


                if i % 50 == 0:
                    cost, rec_x, rec_loss, latent_loss, reg_loss = sess.run(
                        [ops['loss'], ops['rec_x'], ops['rec_loss'], ops['latent_loss'], ops['reg_loss']],
                        feed_dict={ops['x']: X_train_images})
                    print ('step: %04d' % (i), "cost=", "{:.4f}".format(cost))
                    print('Rec loss: %.4f   Latent loss: %.4f  Reg loss: %.4f' % ( rec_loss, latent_loss, reg_loss))
                    plt.subplot(121)
                    plt.imshow(X_train_images[0])
                    plt.subplot(122)
                    plt.imshow(rec_x[0])
                    plt.pause(0.001)

                i += 1

    except tf.errors.OutOfRangeError:
        logging.info('Done training -- epoch limit reached')

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)
    sess.close()



if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    xs,ys = train()

    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111)
    ax.plot(xs,ys)
    ax.set_title('Variational Free Engergy')
    fig_2.savefig('VFE.pdf', bbox_inches='tight')
