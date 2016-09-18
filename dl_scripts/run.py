# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

import siamese
import amos

# train the network!
# references: https://github.com/tensorflow/models/blob/master/inception

## variables
FLAGS = tf.app.flags.FLAGS

# flags regarding data training
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 10.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('batch_size', 10.0,
                          """Epochs after which learning rate decays.""")

# training itself
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_path', '/work/amosexp/data/input/vgg19.npy',
                           """If specified, restore this pretrained model """
                            """before beginning any training.""")

tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")

# some important paths
tf.app.flags.DEFINE_string('data_dir', '/work/amosexp/data/output/', 
                           'Directory for storing data')
tf.app.flags.DEFINE_string('summaries_dir', '/work/amosexp/summaries/', 
                           'Summaries directory')


tf.app.flags.DEFINE_string('tr_dataset', '/datasets/isophia/AMOS/train/', 
                           'Path to train dataset.')
tf.app.flags.DEFINE_string('te_dataset', '/datasets/isophia/AMOS/test/', 
                           'Path to test dataset.')

# total of epochs
tf.app.flags.DEFINE_string('n_epochs_tr', 1000000,
                           """Number of epochs to be realized when training.""")
tf.app.flags.DEFINE_string('n_epochs_te', 1000,
                           """Number of epochs to be realized when testing.""")

# some important (hyper)parameters!
BETA_ONE = 0.9
BETA_TWO = 0.999
EPSILON  = 0.0001

def train(tr_dataset, te_dataset):
    with tf.device('gpu:1'):
        # set config options
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        # initialize session
        sess = tf.InteractiveSession(config = config)

        # summary writers
        tr_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
						sess.graph)
        te_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

        # initialize siamese neural network
        SNN = siamese.siamese(FLAGS.batch_size, FLAGS.pretrained_path)

        # create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus
        global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), 
					  trainable=False)

        decay_steps = int(FLAGS.batch_size * FLAGS.num_epochs_per_decay)

        # decay the learning rate exponentially based on the number of steps
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor)

        # create an optimizer that performs gradient descent
        opt = tf.train.AdamOptimizer(lr)
        opt_op = opt.minimize(SNN.loss)

        saver = tf.train.Saver()
        tf.initialize_all_variables().run()

        for step in range(FLAGS.n_epochs_tr):
            with tf.device('/gpu:1'):
                batch_x1, batch_x2, batch_y = tr_dataset.get_next_batch()

                assert batch_x1 is not None or \
                   batch_x2 is not None or \
                   batch_y is not None, 'Model has reached the end!'

                if step % 100 == 0 and step > 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    _, loss_value = sess.run([opt_op, SNN.loss], feed_dict={
                                             SNN.x1: batch_x1, 
                                             SNN.x2: batch_x2, 
                                             SNN.y:  batch_y},
                                             options      = run_options,
                                             run_metadata = run_metadata)
            
                    tr_writer.add_run_metadata(run_metadata, "%03d" % step)

                    # make sure loss value still fits
                    assert not np.isnan(loss_value.any()), 'Model diverged with loss = NaN'

                else:
                    _, loss_value, loss_summary = sess.run([opt_op, SNN.loss, SNN.loss_sum], 
                                                            feed_dict={
                                                                        SNN.x1: batch_x1, 
                                                                        SNN.x2: batch_x2, 
                                                                        SNN.y:  batch_y})
 
                    # make sure loss value still fits
                    assert not np.isnan(loss_value.any()), 'Model diverged with loss = NaN'
                    print loss_value

                    tr_writer.add_summary(loss_summary, step)

                if step % 5000 == 0 and step > 0:
                    # save current session
                    saver.save(sess, FLAGS.data_dir + "SNN", global_step = step)
                    test(sess, SNN, te_writer, step, te_dataset)

        tr_writer.close()
        te_writer.close()

def test(sess, SNN, te_writer, step, te_dataset):
    for i in range(FLAGS.n_epochs_te):
        batch_x1, batch_x2, batch_y = te_dataset.get_next_batch()

        # test data is done already, go home
        if batch_x1 is None or batch_x2 is None or batch_y is None:
            return

        a1, a2, accuracy_sum = sess.run([SNN.a1, SNN.a2, SNN.accuracy_sum], 
                                         feed_dict = {
                                                      SNN.x1: batch_x1,
                                                      SNN.x2: batch_x2,
                                                      SNN.y:  batch_y })

        te_writer.add_summary(accuracy_sum, step)

if __name__ == "__main__":
    train_dataset = amos.dataset(FLAGS.tr_dataset, FLAGS.batch_size)
    test_dataset = amos.dataset(FLAGS.te_dataset, FLAGS.batch_size)

    # check if there is a valid directory and delete past entries
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)

    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    train(train_dataset, test_dataset)
