# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

import autoencoder
import amos

# train the network!
# references: https://github.com/tensorflow/models/blob/master/inception

## variables
FLAGS = tf.app.flags.FLAGS

# flags regarding data training
tf.app.flags.DEFINE_float('lr', 0.00005,
                          """Initial learning rate.""")

tf.app.flags.DEFINE_integer('batch_size', 15,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_integer('hidden_size', 100,
                          """Size of output of encoder layer.""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")

# some important paths
tf.app.flags.DEFINE_string('data_dir', '/work/amosexp/data/output/', 
                           'Directory for storing data')
tf.app.flags.DEFINE_string('summaries_dir', '/work/amosexp/summaries/', 
                           'Summaries directory')


tf.app.flags.DEFINE_string('tr_dataset', '/datasets/isophia/AMOS/tr_raw_data/pickle_files/', 
                           'Path to train dataset.')
tf.app.flags.DEFINE_string('te_dataset', '/datasets/isophia/AMOS/te_raw_data/pickle_files/', 
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
    with tf.device('/gpu:0'):
        # set config options
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        # initialize session
        sess = tf.InteractiveSession(config = config)
        i_test = 0 # iterator for test evaluation

        # summary writers
        tr_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                      sess.graph)
        te_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

        # initialize VAE neural network
        VAE = VAE.VAE(FLAGS.batch_size, FLAGS.hidden_size)

        # create an optimizer that performs gradient descent
        opt = tf.train.AdamOptimizer(FLAG.lr, beta1=.5)
        opt_op = opt.minimize(VAE.loss, global_step=batch)

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
                batch_x, batch_y = tr_dataset.get_next_batch()

                while batch_x is None or \
                      batch_y is None:
                    batch_x, batch_y = tr_dataset.get_next_batch(restart = True)

                if step % 100 == 0 and step > 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    _, cost = sess.run([opt_op, VAE.loss], feed_dict={
                                             VAE.x: batch_x,
                                             VAE.training: tf.Variable(True, name = 'training'),
                                             VAE.dropout_rate: 0.5},
                                             options      = run_options,
                                             run_metadata = run_metadata)
            
                    tr_writer.add_run_metadata(run_metadata, "%03d" % step)

                    # make sure loss value still fits
                    assert not np.isnan(cost.any()), 'Model diverged with loss = NaN'

                else:
                    _, cost, cost_summary = sess.run([opt_op, VAE.loss, VAE.loss_sum], 
                                                            feed_dict={
                                                                        VAE.x: batch_x,
                                                                        VAE.training: tf.Variable(True, name = 'training'),
                                                                        VAE.dropout_rate: 0.5})
 
                    # make sure loss value still fits
                    assert not np.isnan(cost.any()), 'Model diverged with loss = NaN'
    
                    print "Step %d: " % step
                    print cost

                    tr_writer.add_summary(cost_summary, step)

                if step % 5000 == 0 and step > 0:
                    # save current session
                    saver.save(sess, FLAGS.data_dir + "VAE", global_step = step)
                    
                    # estimate evaluation
                    i_test = test(sess, VAE, te_writer, i_test, te_dataset)

        tr_writer.close()
        te_writer.close()

def test(sess, VAE, te_writer, it, te_dataset):
    for i in range(FLAGS.n_epochs_te):
        batch_x, batch_y = te_dataset.get_next_batch()

        # test data is done already, go home
        while batch_x is None \
           or batch_y is None:
            batch_x, batch_y = te_dataset.get_next_batch(restart = True)

        acc_summary = sess.run([VAE.accuracy_sum], 
                                 feed_dict = {
                                              VAE.x: batch_x,
                                              VAE.y:  batch_y,
                                              VAE.training: tf.Variable(False, name = 'training'),
                                              VAE.dropout_rate: 0.5})

        it += 1

        te_writer.add_summary(acc_summary, it)

    return it

if __name__ == "__main__":
    train_dataset = amos.dataset(FLAGS.tr_dataset, FLAGS.batch_size)
    test_dataset = amos.dataset(FLAGS.te_dataset, FLAGS.batch_size)

    # check if there is a valid directory and delete past entries
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)

    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    train(train_dataset, test_dataset)
