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
tf.app.flags.DEFINE_float('initial_learning_rate', 0.00005,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.96,
                          """Learning rate decay factor.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 1000.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_integer('batch_size', 25,
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
    with tf.device('gpu:0'):
        # set config options
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7

        # initialize session
        sess = tf.InteractiveSession(config = config)
        i_test = 0 # iterator for test evaluation

        # summary writers
        tr_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                      sess.graph)
        te_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

        # initialize siamese neural network
        SNN = siamese.siamese(FLAGS.batch_size, FLAGS.pretrained_path)

        # batch counter
        batch = tf.Variable(0)
        decay_step = int(FLAGS.batch_size * FLAGS.num_epochs_per_decay)
        decay_rate = FLAGS.learning_rate_decay_factor

        # decay the learning rate exponentially based on the number of steps
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                  tf.mul(batch, FLAGS.batch_size),           # current index of dataset
                                  decay_step,                              # decay step
                                  decay_rate)                              # decay rate

        # create an optimizer that performs gradient descent
        opt = tf.train.AdamOptimizer(lr)
        opt_op = opt.minimize(SNN.loss, global_step = batch)

        saver = tf.train.Saver()
        tf.initialize_all_variables().run()

        # check if there is a valid checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.data_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Restoring checkpoint..."

        # iterator for test evaluation
        i_test = 0

        for step in range(FLAGS.n_epochs_tr):
            with tf.device('/gpu:0'):
                batch_x1, batch_x2, batch_y = tr_dataset.get_next_batch()

                while batch_x1 is None or \
                   batch_x2 is None or \
                   batch_y is None:
                    batch_x1, batch_x2, batch_y = tr_dataset.get_next_batch(restart = True)

                if step % 100 == 0 and step > 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    _, loss_value = sess.run([opt_op, SNN.loss], feed_dict={
                                             SNN.x1: batch_x1, 
                                             SNN.x2: batch_x2, 
                                             SNN.y:  batch_y,
                                             SNN.training: tf.Variable(True, name = 'training'),
                                             SNN.dropout_rate: 0.5},
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
                                                                        SNN.y:  batch_y,
                                                                        SNN.training: tf.Variable(True, name = 'training'),
                                                                        SNN.dropout_rate: 0.5})
 
                    # make sure loss value still fits
                    assert not np.isnan(loss_value.any()), 'Model diverged with loss = NaN'
    
                    print "Step %d: " % step
                    print loss_value

                    tr_writer.add_summary(loss_summary, step)

                if step % 5000 == 0 and step > 0:
                    # save current session
                    saver.save(sess, FLAGS.data_dir + "SNN", global_step = step)
                    
                    # estimate evaluation
                    i_test += test(sess, SNN, te_writer, i_test, te_dataset)

        tr_writer.close()
        te_writer.close()

def test(sess, SNN, te_writer, it, te_dataset):
    for i in range(FLAGS.n_epochs_te):
        batch_x1, batch_x2, batch_y = te_dataset.get_next_batch()

        # test data is done already, go home
        while batch_x1 is None \
           or batch_x2 is None \
           or batch_y is None:
            batch_x1, batch_x2, batch_y = te_dataset.get_next_batch(restart = True)

        a1, a2, accuracy_sum = sess.run([SNN.a1, SNN.a2, SNN.accuracy_sum], 
                                         feed_dict = {
                                                      SNN.x1: batch_x1,
                                                      SNN.x2: batch_x2,
                                                      SNN.y:  batch_y,
                                                      SNN.training: tf.Variable(False, name = 'training'),
                                                      SNN.dropout_rate: 1.0})

        it += 1

        te_writer.add_summary(accuracy_sum, it)

    return it

if __name__ == "__main__":
    train_dataset = amos.dataset(FLAGS.tr_dataset, FLAGS.batch_size)
    test_dataset = amos.dataset(FLAGS.te_dataset, FLAGS.batch_size)

    # check if there is a valid directory and delete past entries
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)

    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    train(train_dataset, test_dataset)
