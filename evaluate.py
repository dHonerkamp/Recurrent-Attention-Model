import tensorflow as tf
import logging
from RAM import RAMNetwork
from utility import Utility, auto_adjust_flags
from main import eval
from input_fn import get_data
from Visualization import Visualization


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

    with tf.device('/device:GPU:*'):
        model = RAMNetwork(FLAGS=FLAGS,
                           full_summary=False)

    with tf.Session() as sess:
        model.saver.restore(sess, FLAGS.path + "/cp.ckpt")
        start_step = model.global_step.eval(session=sess)
        tf.logging.info('Evaluate model at step: %d ', start_step)

        train_writer, valid_writer, test_writer, train_handle, valid_handle, test_handle = model.setup(sess, train_data, valid_data, test_data)
        Visual = Visualization(model, FLAGS)

        # Test set
        eval(model, sess, FLAGS, valid_handle, FLAGS.batches_per_eval_valid, valid_writer, prefix='VALIDATION - LAST MODEL: ')
        eval(model, sess, FLAGS, test_handle, FLAGS.batches_per_eval_test, test_writer, prefix='TEST - LAST MODEL: ')
        Visual(sess, 'test_set', test_handle)

        model.saver.restore(sess, FLAGS.path + "/cp_best.ckpt")
        eval(model, sess, FLAGS, test_handle, FLAGS.batches_per_eval_test, test_writer, prefix='TEST - BEST MODEL: ')

        valid_writer.close()
        test_writer.close()