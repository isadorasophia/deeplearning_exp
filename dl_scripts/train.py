# -*- coding: utf-8 -*-

import tensorflow as tf

import siamese
import amos

# train the network!
# references: https://github.com/tensorflow/models/blob/master/inception

## variables
FLAGS = tf.app.flags.FLAGS

# flags regarding data training
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('batch_size', 10.0,
                          """Epochs after which learning rate decays.""")

# training itself
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_path', '../data/vgg19.npy',
                           """If specified, restore this pretrained model """
                            """before beginning any training.""")

tf.app.flags.DEFINE_integer('num_gpus', 0,
                            """How many GPUs to use.""")

# some important paths
tf.app.flags.DEFINE_string('train_dir', '/media/bonnibel/Jerônimo/AMOS_Data/dataset/evaluation/00000016/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('test_dir', '/tmp/amos_test',
                           """Directory where to write event logs """
                           """and checkpoint.""")

# total of epochs
tf.app.flags.DEFINE_string('n_epochs', 300000,
                           """Number of epochs to be realized.""")

# some important (hyper)parameters!
MOMENTUM = 0.9
EPSILON = 1.0

def train(dataset):
    # initialize session
    sess = tf.InteractiveSession()

    # initialize siamese neural network
    SNN = siamese.siamese(FLAGS.batch_size, FLAGS.pretrained_path);

    # create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus
    global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), 
                                  trainable=False)

    decay_steps = int(FLAGS.batch_size * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor)

    # create an optimizer that performs gradient descent
    opt = tf.train.AdamOptimizer(lr,
                                 momentum=MOMENTUM,
                                 epsilon=EPSILON)

    opt_op = opt.minimize(SNN.loss)

    saver = tf.train.Saver()
    tf.initialize_all_variables().run()

    for step in range(FLAGS.n_epochs):
        batch_x1, batch_x2, batch_y = amos.train.next_batch(FLAGS.batch_size)

        _, loss_value = sess.run([opt_op, SNN.loss], feed_dict={
                                 SNN.x1: batch_x1, 
                                 SNN.x2: batch_x2, 
                                 SNN.y_: batch_y})

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5000 == 0 and step > 0:
            saver.save(sess, 'model.ckpt')

if __name__ == "__main__":
    train_dataset = amos.dataset(FLAGS.train_dir, FLAGS.batch_size)

    train(train_dataset)
