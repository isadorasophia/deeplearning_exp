import tensorflow as tf
import numpy as np

# Implementation of a siamese model, relying on the VGG19 net architecture;
# source: [https://arxiv.org/pdf/1409.1556.pdf]
#
# references: https://github.com/jazzsaxmafia/Weakly_detector/blob/master/src/detector.py;
#             https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py

IMG_DIM  = 224
VGG_MEAN = [103.939, 116.779, 123.68]

class siamese:
    def __init__(self, batch_size, weight_path):
        # input for first and second network
        self.x1 = tf.placeholder(tf.float32, shape = (batch_size, 3, IMG_DIM, IMG_DIM), name = "x1")
        self.x2 = tf.placeholder(tf.float32, shape = (batch_size, 3, IMG_DIM, IMG_DIM), name = "x2")

        # pre-trained weights of siamese model, according to VGG 19
        self.s_dict = np.load(weight_path, encoding='latin1').item()

        # image mean values from VGG model
        self.image_mean = [103.939, 116.779, 123.68]

        # activation values, i.e. output
        with tf.variable_scope("siamese") as scope:
            self.a1 = self.network(self.x1)

            scope.reuse_variables()

            self.a2 = self.network(self.x2)

        # label, a.k.a. correct output
        self.y = tf.placeholder(tf.int32, shape = (batch_size, 1), name = "y")

        # estimate loss
        self.loss = self.loss()
        self.accuracy = self.accuracy()

        # dropout porpuses
        # self.keep_prob = tf.placeholder("float", name = "dropout_keep_prob")

    def network(self, x):
        # receives x := BGR image of size [batch, 3, height, width]

        # split it into different tensors
        pre_x = tf.transpose(x, [0, 3, 2, 1])
        b, g, r = tf.split(3, 3, pre_x)

        # takes the mean value
        bgr = tf.concat(3, [b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2], ])

        # make sure final shape meets the shape
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        conv1_1 = self.conv_layer(bgr, "conv1_1")              # 224 x 224 x 64
        conv1_2 = self.conv_layer(conv1_1, "conv1_2") 
        pool1 = self.max_pool(conv1_2, "pool1")                # 112 x 112 x 128

        conv2_1 = self.conv_layer(pool1, "conv2_1")            # 112 x 112 x 128
        conv2_2 = self.conv_layer(conv2_1, "conv2_2") 
        pool2 = self.max_pool(conv2_2, "pool2")                # 56 x 56 x 256

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2") 
        conv3_3 = self.conv_layer(conv3_2, "conv3_3") 
        conv3_4 = self.conv_layer(conv3_3, "conv3_4")
        pool3 = self.max_pool(conv3_4, "pool3")                # 28 x 28 x 512

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        conv4_4 = self.conv_layer(conv4_3, "conv4_4")
        pool4 = self.max_pool(conv4_4, "pool4")                # 14 x 14 x 512

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        conv5_4 = self.conv_layer(conv5_3, "conv5_4")
        pool5 = self.max_pool(conv5_4, "pool5")                # 7 x 7 x 512

        # fully connected
        fc6 = self.fc_layer(pool5, "fc6")                      # 1 x 1 x 4096
        assert fc6.get_shape().as_list()[1:] == [4096]         # make sure it fits
        relu6 = tf.nn.relu(fc6)

        fc7   = self.fc_layer(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)                                # 1 x 1 x 1000

        fc8   = self.fc_layer(relu7, "fc8") 
        relu8 = tf.nn.relu(fc8)

        prob  = self.new_fc_layer(relu8, 1000, 1, "prob")

        s_dict = None

        return prob

    # take the average pooling of the result, aka the size turns into half
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name, reuse = None) as scope:
            filt = self.get_filter(name)
            b = self.get_bias(name)

            # initialize values
            conv_filt = tf.get_variable(
                        "W",
                        initializer = filt
                        )

            conv_bias = tf.get_variable(
                        "b",
                        initializer = b
                        )

            conv = tf.nn.conv2d(bottom, conv_filt, [1, 1, 1, 1], padding = 'SAME')
            bias = tf.nn.bias_add(conv, conv_bias)
            relu = tf.nn.relu(bias, name = name)

        return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name, reuse = None):
            shape = bottom.get_shape().as_list()
            dim = np.prod(shape[1:])

            # flatten x
            x = tf.reshape(bottom, [-1, dim])

            w = self.get_fc_weight(name)
            b = self.get_bias(name)

            weights = tf.get_variable(
                      "W",
                      initializer = w
                      )

            bias    = tf.get_variable(
                      "b",
                      initializer = b
                      )

            fc = tf.nn.bias_add(tf.matmul(x, weights), bias, name = name)

        return fc

    def new_fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name) as scope:
            # flatten x
            x = tf.reshape(bottom, [-1, in_size])

            weights = tf.get_variable(
                      "W",
                      shape = [in_size, out_size],
                      initializer = tf.random_normal_initializer(0., 0.01)
                      )

            bias = tf.get_variable(
                   "b",
                   shape = [out_size],
                   initializer = tf.constant_initializer(0.)
                   )

            fc = tf.nn.bias_add(tf.matmul(x, weights), bias, name = name)

        return fc

    # get weights according to the layer
    def get_filter(self, name):
        return tf.constant(self.s_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.s_dict[name][1], name="biases") 

    def get_fc_weight(self, name):
        return tf.constant(self.s_dict[name][0], name="weights")

    def loss(self):
        # Y:     if x1 is older
        labels_o = tf.cast(self.y, tf.float32)

        # 1 - Y: if x1 is newer
        labels_n = tf.cast(tf.sub(1, self.y, name="oneSubYi"), tf.float32)

        # L1 normalization
        # E_w = tf.reduce_mean(tf.abs(tf.sub(self.a1, self.a2)), 1, keep_dims = True)

        # L2 normalization
        E_w = tf.nn.l2_normalize(tf.sub(self.a1, self.a2), 1)

        Q = tf.reduce_max(E_w, keep_dims = True)

        loss = tf.add(tf.mul(2/Q, tf.mul(labels_n, tf.pow(E_w, 2))),
                      tf.mul(2*Q, tf.mul(labels_o,
                                         tf.pow (np.e, tf.mul(-2.77/Q, E_w))
                     )))

        return loss

    def accuracy(self):
        # estimate result
        res = tf.nn.l2_normalize(tf.sub(self.a1, self.a2), 1)

        correct_prediction = tf.equal(res, tf.cast(self.y, tf.float32))

        final_ac = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        return final_ac
