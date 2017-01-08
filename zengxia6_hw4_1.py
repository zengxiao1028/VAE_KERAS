import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
tf.set_random_seed(0)

input_dim = 3
batch_size = 256
latent_dim = 2

def gen_swiss_roll():
    a_and_b = np.random.multivariate_normal([10, -10], [[10, -5], [-5, 10]], size=(batch_size))
    swiss_roll = [( each[0]*np.cos(each[0]),each[1], each[0]*np.sin(each[0]) ) for each in a_and_b]
    return np.array(swiss_roll)

def create_network():
    #input
    x = tf.placeholder(tf.float32, [None, input_dim])

    with slim.arg_scope([slim.convolution2d]):
        #encoder
        conv1 = slim.repeat(x, 2, slim.conv2d, 32, 3, scope='conv1')
        pool1 = slim.max_pool2d(conv1, 3)

        conv2 = slim.repeat(pool1,2,slim.conv2d,32,3,scope='conv2')
        pool2 = slim.max_pool2d(conv2,3)

        conv3 = slim.repeat(pool2,2,slim.conv2d,32,3,scope='conv3')
        conv3_flat = slim.flatten(conv3)

        fc1 = slim.fully_connected(conv3_flat, 32)
        encoder_fc = slim.fully_connected(fc1, 10, activation_fn=None)

        encoder_mu = slim.fully_connected(encoder_fc, latent_dim, scope='encoder_mu', activation_fn=None)
        encoder_sigma_square = tf.square(slim.fully_connected(encoder_fc, latent_dim, scope='encoder_sigma',
                                                              activation_fn=None))
        encoder_sigma = tf.sqrt(encoder_sigma_square)

        #sample z
        eps = tf.random_normal((batch_size, latent_dim), 0, 1,dtype=tf.float32)
        z = tf.add(encoder_mu,tf.mul(encoder_sigma, eps))

        #decoder
        conv1 = slim.repeat(x, 2, slim.conv2d, 32, 3, scope='conv1')
        decoder_fc =  slim.stack(z, slim.fully_connected, [32, 32, 32,32], scope='decoder')
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
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.view_init(15, 95)
    plt.show()
    ax1.set_xlim(-25,25)
    ax1.set_ylim(-25,25)
    ax1.set_zlim(-25,25)
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.view_init(15, 95)
    plt.show()
    ax2.set_xlim(-25,25)
    ax2.set_ylim(-25,25)
    ax2.set_zlim(-25,25)

    xs = []
    ys = []
    for i in range(20000):
        batch_xs  = gen_swiss_roll()
        # Fit training using batch data

        _ ,cost,rec_x,rec_loss,latent_loss = sess.run([optimizer,ops['loss'],ops['rec_x'], ops['rec_loss'],ops['latent_loss']],feed_dict={ops['x']:batch_xs})

        if i % 20 == 0 :
            xs.append(i)
            ys.append(cost)

        if i % 500 == 0:
            print ( 'step: %04d' %  (i), "cost=", "{:.4f}".format(cost))
            print('Rec loss: %.4f   Latent loss: %.4f' % (np.mean(rec_loss),np.mean(latent_loss)))
            ax1.clear()
            ax1.set_xlim(-25, 25)
            ax1.set_ylim(-25, 25)
            ax1.set_zlim(-25, 25)
            ax1.set_title('Real')
            ax1.scatter(batch_xs[:,0] ,batch_xs[:,1] ,batch_xs[:,2],c='red')
            ax2.clear()
            ax2.set_xlim(-25, 25)
            ax2.set_ylim(-25, 25)
            ax2.set_zlim(-25, 25)
            ax2.set_title('Generated')
            ax2.scatter(rec_x[:, 0], rec_x[:, 1], rec_x[:, 2],c='blue')
            plt.pause(0.001)

    fig.savefig('swissroll.pdf', bbox_inches='tight')
    return xs, ys


if __name__=='__main__':

    xs,ys = train()

    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111)
    ax.plot(xs,ys)
    ax.set_title('Variational Free Engergy')
    fig_2.savefig('VFE.pdf', bbox_inches='tight')
