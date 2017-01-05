import tensorflow as tf
import numpy as np

# Implementation of an auto-encoding variational bayes;
# source: [https://arxiv.org/pdf/1312.6114v10.pdf]
#
# references: https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW/blob/master/main-vae.py;

IMG_DIM = 224

class VAE:
    def __init__(self, batch_size, z_dim):
        # input for encoder network
        self.x = tf.placeholder(tf.float16, \
                                shape = (batch_size, IMG_DIM, IMG_DIM, 3), name="x")

        # # label, a.k.a. correct output
        # self.y = tf.placeholder(tf.int16, shape = (batch_size, 1), name="y")

        self.batch_size = batch_size
        self.z_dim = z_dim

        # is this a training session?
        self.training = tf.Variable(True, name='training')

        # define dropout rate
        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

        self.z, self.mean, self.std = self.encoder(self.x)
        self.y = self.decoder(self.z)

        # estimate loss
        self.loss = self.loss(self.mean, self.epsilon, self.std)

        # debug visualization
        self.loss_sum = tf.scalar_summary("loss", tf.reduce_mean(self.loss))

    def encoder(self, x):
        # x := HSV image of dimensions [batch, height, width, 3]
        conv1 = self.conv_layer(x, [4, 4, 3, 64], "conv1", batch_norm=False)
        conv2 = self.conv_layer(conv1, [4, 4,  64, 128], "conv2")
        conv3 = self.conv_layer(conv2, [4, 4, 128, 256], "conv3")
        conv4 = self.conv_layer(conv3, [4, 4, 256, 512], "conv4")
        conv5 = self.conv_layer(conv4, [4, 4, 512, 512], "conv5")
        conv6 = self.conv_layer(conv5, [4, 4, 512, 512], "conv6")
        conv7 = self.conv_layer(conv6, [4, 4, 512, 512], "conv7")
        conv8 = self.conv_layer(conv7, [4, 4, 512, 512], "conv8")

        # fully connected
        mean = self.fc_layer(conv8, 4096, self.z_dim, "mean")

        # apply normal distribution and std from mean value
        epsilon = tf.random_normal([self.batch_size, self.z_dim])
        std = tf.sqrt(tf.exp(mean))

        # find out z
        z = tf.add(mean, tf.multiply(std, epsilon))

        return z, mean, std

    def decoder(self, z):
        # z := feature of dimensions [batch, z_dim, 1]
        conv1 = self.conv_layer(z, [4, 4, 1, 512], "conv1", dropout=True)
        conv2 = self.conv_layer(conv1, [4, 4, 512, 512], "conv2", dropout=True)
        conv3 = self.conv_layer(conv2, [4, 4, 512, 512], "conv3", dropout=True)
        conv4 = self.conv_layer(conv3, [4, 4, 512, 512], "conv4")
        conv5 = self.conv_layer(conv4, [4, 4, 512, 512], "conv5")
        conv6 = self.conv_layer(conv5, [4, 4, 512, 256], "conv6")
        conv7 = self.conv_layer(conv6, [4, 4, 256, 128], "conv7")
        conv8 = self.conv_layer(conv7, [4, 4, 128, 64], "conv8")
        y = self.conv_layer(conv8, [4, 4, 64, 3], "y", batch_norm=False)

        return y

    def conv_layer(self, bottom, dim, name, batch_norm=True, dropout=False):
        with tf.variable_scope(name, reuse = None) as scope:
            # initialize values
            conv_filt = tf.get_variable(
                                        "W",
                                        shape = dim,
                                        initializer = tf.contrib.layers.xavier_initializer_conv2d()
                                        )
            conv_bias = tf.get_variable(
                                        "b",
                                        shape = [dim[-1]],
                                        initializer = tf.constant_initializer(0.)
                                        )

            conv = tf.nn.conv2d(bottom, filter=conv_filt, stride=[1, 2, 2, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_bias)

            # apply batch norm
            if batch_norm:
                bias = tf.contrib.layers.batch_norm(bias, is_training=self.training)

            # apply dropout
            if dropout:
                bias = tf.nn.dropout(bias, self.dropout_rate)

            relu = tf.nn.elu(bias, name=name)

        return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name) as scope:
            # flatten x
            x = tf.reshape(bottom, [-1, in_size])

            weights = tf.get_variable(
                      "W",
                      shape = [in_size, out_size],
                      initializer = tf.contrib.layers.xavier_initializer()
                      )

            bias = tf.get_variable(
                   "b",
                   shape = [out_size],
                   initializer = tf.constant_initializer(0.)
                   )

            fc = tf.nn.bias_add(tf.matmul(x, weights), bias, name = name)
            dp = tf.nn.dropout(fc, self.dropout_rate)

        return dp

    def loss(self, mean, std, epsilon=1e-6):
        reconstruction_loss = -tf.reduce_sum(self.x * tf.log(self.y+epsilon) +
                                             (1.0 - self.x) * 
                                             tf.log(1.0 - output_tensor + epsilon))

        latent_loss = .5 * tf.reduce_sum(2.0 * tf.log(std + epsilon) - 
                                         tf.square(mean) - tf.square(std) + 1.0)

        cost = tf.reduce_mean(reconstruction_loss, latent_loss)

        return cost

