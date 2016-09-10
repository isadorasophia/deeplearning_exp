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
        self.x1 = tf.placeholder(tf.uint8, shape = (batch_size, IMG_DIM), name = "x1")
        self.x2 = tf.placeholder(tf.uint8, shape = (batch_size, IMG_DIM), name = "x2")

        # pre-trained weights of siamese model, according to VGG 19
        self.s_dict = np.load(weight_path, encoding='latin1').item()

        # image mean values from VGG model
        self.image_mean = [103.939, 116.779, 123.68]

        # activation values, i.e. output
        with tf.variable_scope("siamese") as scope:
            self.a1 = self.network(x1)

            scope.reuse_variables()

            self.a2 = self.network(x2)

        # label, a.k.a. correct output
        self.y = tf.placeholder(tf.uint8, shape = (batch_size), name = "y")

        # estimate loss
        self.loss = self.loss()

        # dropout porpuses
        # self.keep_prob = tf.placeholder("float", name = "dropout_keep_prob")

    def network(self, x):
        # receives x := BGR image of size [batch, 3, height, width]

        # split it into different tensors
        pre_x = np.swapaxes(np.swapaxes(x, 1, 3), 1, 2)
        b, g, r = tf.split(3, 3, pre_x)

        # takes the mean value
        bgr = tf.concat(3, [blue - VGG_MEAN[0], green - VGG_MEAN[1],
                            red - VGG_MEAN[2]])

        # make sure final shape meets the shape
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]


        ####### HINT: use variable scope reuse = True
        # http://stackoverflow.com/questions/36844909/siamese-neural-network-in-tensorflow/36847703#36847703
        conv1_11 = self.conv_layer(bgr, "conv1_1")          # 224 x 224 x 64
        conv1_21 = self.conv_layer(conv1_11, "conv1_2") 
        pool11 = self.max_pool(conv1_21, "pool1")       # 112 x 112 x 128

        conv2_11 = self.conv_layer(pool11, "conv2_1")   # 112 x 112 x 128
        conv2_21 = self.conv_layer(conv2_11, "conv2_2") 
        pool21 = self.max_pool(conv2_21, "pool2")       # 56 x 56 x 256

        conv3_11 = self.conv_layer(pool21, "conv3_1")
        conv3_21 = self.conv_layer(conv3_11, "conv3_2") 
        conv3_31 = self.conv_layer(conv3_21, "conv3_3") 
        conv3_41 = self.conv_layer(conv3_31, "conv3_4")
        pool31 = self.max_pool(conv3_41, "pool3")       # 28 x 28 x 512

        conv4_11 = self.conv_layer(pool31, "conv4_1")
        conv4_21 = self.conv_layer(conv4_11, "conv4_2")
        conv4_31 = self.conv_layer(conv4_21, "conv4_3")
        conv4_41 = self.conv_layer(conv4_31, "conv4_4")
        pool41 = self.max_pool(conv4_41, "pool4")       # 14 x 14 x 512

        conv5_11 = self.conv_layer(pool41, "conv5_1")
        conv5_21 = self.conv_layer(conv5_11, "conv5_2")
        conv5_31 = self.conv_layer(conv5_21, "conv5_3")
        conv5_41 = self.conv_layer(conv5_31, "conv5_4")
        pool51 = self.max_pool(conv5_41, "pool5")       # 7 x 7 x 512

        # fully connected
        fc61 = self.fc_layer(pool51, "fc6")             # 1 x 1 x 4096
        assert self.fc6.get_shape().as_list()[1:] == [4096]     # make sure it fits
        relu61 = tf.nn.relu(fc61)

        fc71   = self.fc_layer(relu61, "fc7")
        relu71 = tf.nn.relu(fc71)                       # 1 x 1 x 1000

        fc81 = fc_layer(relu71, "fc8")   
        prob = tf.nn.softmax(fc8_drop, name = "prob")

        s_dict = None

        return prob

    # take the average pooling of the result, aka the size turns into half
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize = [1, 2, 2, 1], stides = [1, 2, 2, 1], padding = 'SAME', name = name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name, reuse = True):
            filt = self.get_filter(name)
            b = self.get_bias(name)

            # initialize values
            conv_filt = tf.get_variable(
                        "W",
                        shape = filt.shape,
                        initializer = tf.constant_initializer(filt)
                        )

            conv_bias = tf.get_variable(
                        "b",
                        shape = b.shape,
                        initializer = tf.constant_initializer(b)
                        )

            conv = tf.nn.conv2d(bottom, conv_filt, [1, 1, 1, 1], padding = 'SAME')
            bias = tf.nn.bias_add(conv, conv_bias)
            relu = tf.nn.relu(bias, name = name)

        return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name, reuse = True):
            shape = bottom.get_shape().as_list()
            dim = np.prod(shape[1:])

            # flatten x
            x = tf.reshape(bottom, [-1, dim])

            w = self.get_fc_weight(name)
            b = self.get_bias(name)

            weights = tf.get_variable(
                      "W",
                      shape=cw.shape,
                      initializer=tf.constant_initializer(w))
            bias    = tf.get_variable(
                      "b",
                      shape=b.shape,
                      initializer=tf.constant_initializer(b))

            fc = tf.nn.bias_add(tf.matmul(x, weights), bias, name=scope)

        return fc

    # get weights according to the layer
    def get_filter(self, name):
        return tf.constant(self.s_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.s_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.s_dict[name][0], name="weights")

    def loss(self):
        # Y
        # if x1 is older
        labels_o = self.y

        # 1 - Y
        # if x1 is newer
        labels_n = tf.sub(1, self.y, name="1 - yi")

        E_w = tf.abs((tf.sub(self.o1, self.o2)))
        Q = 2

        loss = tf.scalar_mul(2/Q, tf.mul(labels_n, tf.pow(E_w, 2))) +
               tf.scalar_mul(2 * Q, tf.mul(labels_o, 
                             tf.pow(np.e, tf.scalar_mul(-2.77/Q, E_w))))

        return loss

