import tensorflow as tf
import logging
from RAM import RAMNetwork
from utility import Utility, auto_adjust_flags
from main import store_visualization, eval
from input_fn import get_data
from matplotlib import pyplot as plt
import numpy as np
import time


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # Parsing experimental set up
    FLAGS, _ = Utility.parse_arg()
    # set img_shape, padding, num_classes according to dataset (ignoring cl inputs!)
    auto_adjust_flags(FLAGS)

    if not FLAGS.start_checkpoint:
        raise ValueError('NO MODEL TO LOAD SPECIFIED')

    experiment_name = FLAGS.start_checkpoint
    t_sz = (str(FLAGS.translated_size) if FLAGS.translated_size else "")

    FLAGS.path = FLAGS.summaries_dir + '/' + FLAGS.dataset + t_sz + '/' + experiment_name
    logging.info('\nPATH: ' + FLAGS.path + '\nMODEL: ' + experiment_name + '\n')

    # load datasets
    train_data, valid_data, test_data = get_data(FLAGS)
    FLAGS.train_data_shape = (train_data[0].shape, train_data[1].shape)
    FLAGS.valid_data_shape = (valid_data[0].shape, valid_data[1].shape)
    FLAGS.test_data_shape = (test_data[0].shape, test_data[1].shape)
    FLAGS.data_dtype = (train_data[0].dtype, train_data[1].dtype)

    with tf.device('/device:GPU:*'):
        model = RAMNetwork(FLAGS=FLAGS,
                         patch_shape=len(FLAGS.scale_sizes) * np.power(FLAGS.scale_sizes[0], 2) * FLAGS.img_shape[-1],
                         num_glimpses=FLAGS.num_glimpses,
                         batch_size=FLAGS.batch_size * FLAGS.MC_samples,
                         full_summary=False)


    with tf.Session() as sess:
        model.saver.restore(sess, FLAGS.path + "/cp.ckpt")
        start_step = model.global_step.eval(session=sess)
        tf.logging.info('Evaluate model at step: %d ', start_step)

        # train_writer = tf.summary.FileWriter(FLAGS.path + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.path + '/valid')
        test_writer = tf.summary.FileWriter(FLAGS.path + '/test')

        train_handle = sess.run(model.train_init_op.string_handle())
        valid_handle = sess.run(model.valid_init_op.string_handle())
        test_handle = sess.run(model.test_init_op.string_handle())
        sess.run(model.train_init_op.initializer, feed_dict={model.features_ph_train: train_data[0],
                                                             model.labels_ph_train: train_data[1]})
        sess.run(model.valid_init_op.initializer, feed_dict={model.features_ph_valid: valid_data[0],
                                                             model.labels_ph_valid: valid_data[1]})
        sess.run(model.test_init_op.initializer, feed_dict={model.features_ph_test: test_data[0],
                                                            model.labels_ph_test: test_data[1]})

        # Test set
        eval(model, sess, FLAGS, valid_handle, FLAGS.batches_per_eval_valid, valid_writer, prefix='VALIDATION - LAST MODEL: ')
        eval(model, sess, FLAGS, test_handle, FLAGS.batches_per_eval_test, test_writer, prefix='TEST - LAST MODEL: ')
        store_visualization(sess, model, 'test_set', test_handle, FLAGS)

        model.saver.restore(sess, FLAGS.path + "/cp_best.ckpt")
        eval(model, sess, FLAGS, test_handle, FLAGS.batches_per_eval_test, test_writer, prefix='TEST - BEST MODEL: ')

        valid_writer.close()
        test_writer.close()