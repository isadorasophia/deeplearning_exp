import tensorflow as tf
import numpy as np

# Implementation of a siamese model, relying on the VGG19 net architecture;
# source: [https://arxiv.org/pdf/1409.1556.pdf]
#
# references: https://github.com/jazzsaxmafia/Weakly_detector/blob/master/src/detector.py;
#             https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py

IMG_DIM  = 224

class siamese:
    def __init__(self, batch_size, weight_path):
        # input for first and second network
        self.x1 = tf.placeholder(tf.float16, \
                                 shape = (batch_size, IMG_DIM, IMG_DIM, 3), name = "x1")
        self.x2 = tf.placeholder(tf.float16, \
                                 shape = (batch_size, IMG_DIM, IMG_DIM, 3), name = "x2")

        # pre-trained weights of siamese model, according to VGG 19
        self.s_dict = np.load(weight_path, encoding='latin1').item()

        # image mean values from AMOS dataset, in HSV
        self.img_mean = [0, 2.1, 36.9] # RGB: [93.689, 91.849, 92.119]

        # is this a training session?
        self.training = tf.Variable(True, name='training')

        # define dropout rate
        self.dropout_rate = tf.placeholder(tf.float32, name = "dropout_rate")

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

        self.accuracy_sum = tf.scalar_summary("accuracy", self.accuracy)
        self.loss_sum = tf.scalar_summary("loss", tf.reduce_mean(self.loss))

    def network(self, x):
        # receives x := HSV image of size [batch, 3, height, width]

        # split it into different tensors
        # pre_x = tf.transpose(x, [0, 3, 2, 1])
        h, s, v = tf.split(3, 3, x)

        # takes the mean value
        bgr = tf.concat(3, [h - self.img_mean[0], s - self.img_mean[1], \
                            v - self.img_mean[2], ])

        # make sure final shape meets the shape
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        conv1_1 = self.rand_conv_layer(bgr, "conv1_1")         # 224 x 224 x 64
        conv1_2 = self.rand_conv_layer(conv1_1, "conv1_2") 
        pool1 = self.max_pool(conv1_2, "pool1")                # 112 x 112 x 128

        conv2_1 = self.rand_conv_layer(pool1, "conv2_1")       # 112 x 112 x 128
        conv2_2 = self.rand_conv_layer(conv2_1, "conv2_2") 
        pool2 = self.max_pool(conv2_2, "pool2")                # 56 x 56 x 256

        conv3_1 = self.rand_conv_layer(pool2, "conv3_1")
        conv3_2 = self.rand_conv_layer(conv3_1, "conv3_2") 
        conv3_3 = self.rand_conv_layer(conv3_2, "conv3_3") 
        conv3_4 = self.rand_conv_layer(conv3_3, "conv3_4")
        pool3 = self.max_pool(conv3_4, "pool3")                # 28 x 28 x 512

        conv4_1 = self.rand_conv_layer(pool3, "conv4_1")
        conv4_2 = self.rand_conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.rand_conv_layer(conv4_2, "conv4_3")
        conv4_4 = self.rand_conv_layer(conv4_3, "conv4_4")
        pool4 = self.max_pool(conv4_4, "pool4")                # 14 x 14 x 512

        conv5_1 = self.rand_conv_layer(pool4, "conv5_1")
        conv5_2 = self.rand_conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.rand_conv_layer(conv5_2, "conv5_3")
        conv5_4 = self.rand_conv_layer(conv5_3, "conv5_4")
        pool5 = self.max_pool(conv5_4, "pool5")                # 7 x 7 x 512

        # fully connected
        fc6 = self.new_fc_layer(pool5, 25088, 4096, "fc6")     # 1 x 1 x 4096   
        relu6 = tf.nn.relu(fc6)
        drop6 = tf.nn.dropout(relu6, self.dropout_rate)

        fc7   = self.new_fc_layer(drop6, 4096, 4096, "fc7")    # 1 x 1 x 4096
        relu7 = tf.nn.relu(fc7)                                
        drop7 = tf.nn.dropout(relu7, self.dropout_rate)

        fc8   = self.new_fc_layer(drop7, 4096, 1, "fc8")       # 1 x 1 x 1
        relu8  = tf.nn.relu(fc8)

        prob = tf.nn.dropout(relu8, self.dropout_rate)         # final result

        s_dict = None

        return prob

    # take the average pooling of the result, aka the size turns into half
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize = [1, 2, 2, 1], \
                              strides = [1, 2, 2, 1], padding = 'SAME', name = name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize = [1, 2, 2, 1], \
                              strides = [1, 2, 2, 1], padding = 'SAME', name = name)

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

            # apply batch norm
            norm = tf.contrib.layers.batch_norm(bias, is_training = self.training)

            relu = tf.nn.relu(norm, name = name)

        return relu

    def rand_conv_layer(self, bottom, name, recover_shape = True):
        with tf.variable_scope(name, reuse = None) as scope:
            filt = self.get_filter(name)
            b    = self.get_bias(name)

            # initialize values
            conv_filt = tf.get_variable(
                        "W",
                        shape = filt.get_shape(),
                        initializer = tf.contrib.layers.xavier_initializer_conv2d()
                        )
            conv_bias = tf.get_variable(
                        "b",
                        shape = b.get_shape(),
                        initializer = tf.constant_initializer(0.)
                        )

            conv = tf.nn.conv2d(bottom, conv_filt, [1, 1, 1, 1], padding = 'SAME')
            bias = tf.nn.bias_add(conv, conv_bias)

            # apply batch norm
            norm = tf.contrib.layers.batch_norm(bias, is_training = self.training)

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
                      initializer = tf.contrib.layers.xavier_initializer()
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
        ## implements loss function from S. Chopra, R. Hadsell and Y. LeCun,
        ##                        "Learning a Similarity Metric Discriminatively, 
        ##                                 with Application to Face Verification"
        ##
        # Y:     if x1 is older
        # labels_o = tf.cast(self.y, tf.float32)

        # 1 - Y: if x1 is newer
        # labels_n = tf.cast(tf.sub(1, self.y, name="oneSubYi"), tf.float32)

        # L1 normalization!
        # E_w = tf.reduce_sum(tf.abs(tf.sub(self.a1, self.a2)), 1, keep_dims = True)

        # L2 normalization
        # E_w = tf.nn.l2_normalize(tf.sub(self.a1, self.a2), 1)

        # Q = tf.cast(10, tf.float32)

        # loss = tf.add(tf.mul(2/Q, tf.mul(labels_n, tf.pow(E_w, 2))),
        #               tf.mul(2*Q, tf.mul(labels_o,
        #                                  tf.pow (np.e, tf.mul(-2.77/Q, E_w))
        #               )))

        ## apply l2 norm raw value
        ##
        # d1 = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(self.a1, self.a2)),
        #                            reduction_indices=1) 
        #             )

        ## apply l2 norm distance and apply cosine distance
        ##
        # p1 = tf.nn.l2_normalize(self.a1, dim=1)
        # p2 = tf.nn.l2_normalize(self.a2, dim=1)

        # cos_dis = tf.matmul(normed_array, 
        #                     tf.transpose(normed_embedding, [1, 0]))

        ### apply new loss function!
        p1 = self.a1
        p2 = self.a2

        # get distances from right answer
        d1 = tf.maximum(tf.sub(tf.abs(p1), tf.abs(p2)), 0)
        d2 = tf.maximum(tf.sub(tf.abs(p2), tf.abs(p1)), 0)

        # Y:     if x1 is older
        label_o = tf.cast(self.y, tf.float32)

        # 1 - Y: if x1 is newer
        label_n = tf.cast(tf.sub(1, self.y), tf.float32)

        # constant
        c = tf.cast(1/2, tf.float32)
 
        # apply loss
        loss = tf.mul(c, tf.add(tf.mul(label_o, d1), tf.mul(label_n, d2)
                               )
                     )

        return loss

    def accuracy(self):
        # estimate result
        # res = tf.nn.l2_normalize(tf.sub(self.a1, self.a2), 1)

        # estimate result based on l1 norm
        # res = tf.reduce_sum(tf.abs(tf.sub(self.a1, self.a2)), 1, keep_dims = True)

        ### subtract and check if the result was either positive and negative:
        ###   and begin to work from that!
        p1 = self.a1
        p2 = self.a2

        res = tf.sub(p1, p2)
        
        f1 = lambda x: tf.constant(1.0) if tf.less(res, 0) is True else tf.constant(0.0)

        res = tf.map_fn(f1, res)

        correct_prediction = tf.equal(res, tf.cast(self.y, tf.float32))

        final_ac = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
        return final_ac

