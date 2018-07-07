import tensorflow as tf
import logging
from tqdm import tqdm
import numpy as np
import time
import os
from RAM import RAMNetwork
from input_fn import get_data
from utility import Utility, auto_adjust_flags
from Visualization import Visualization


def eval(model, sess, FLAGS, handle, num_batches, writer, prefix, is_training=False):
    '''
    :param dataset: tuple of (init_op, data, placeholder)
    '''
    fetch = [model.summary, model.global_step, model.accuracy, model.accuracy_MC, model.reward, model.loss, model.xent,
             model.RL_loss, model.baselines_mse, model.learning_rate, model.unknown_accuracy]

    vars = ['acc', 'acc_MC', 'reward', 'loss', 'xent', 'RL_loss', 'b_mse', 'lr', 'acc_unknown']
    averages = np.zeros(len(vars))
    assert(len(fetch) - 2 == len(vars))

    for _ in range(num_batches):
        out = sess.run(fetch, feed_dict={model.is_training: is_training,
                                         model.handle: handle})
        averages += np.array(out[2:]) / num_batches

    summs = [tf.Summary.Value(tag="batch/" + var, simple_value=avg) for var, avg in zip(vars, averages)]
    batch_values = tf.Summary(value=summs)
    step = out[1]
    writer.add_summary(batch_values, step)
    writer.add_summary(out[0], step)

    strs = [item for pair in zip(vars, averages) for item in pair]
    s = 'step ' + str(step) + ' - ' + prefix + len(vars) * ' {}: {:.3f}'
    logging.info(s.format(*strs))

    report = dict(zip(vars, averages))
    report['step'] = step
    report['MC_sampling'] = FLAGS.MC_samples
    report['loc_std'] = FLAGS.loc_std

    return report


def train_model(model, FLAGS):

    with tf.Session() as sess:

        train_writer, valid_writer, test_writer, train_handle, valid_handle, test_handle = model.setup(sess, train_data, valid_data, test_data)

        Visual = Visualization(model, FLAGS)
        epochs_completed = 0
        best_acc_validation = 0.

        print("t - before train: {:.2f}s".format(time.time() - start_time))
        while epochs_completed < FLAGS.num_epochs:

            # Evaluate and visualize
            if epochs_completed % FLAGS.eval_step_interval == 0:
                Visual(sess, 'epoch_{}'.format(epochs_completed), valid_handle)

                _ = eval(model, sess, FLAGS, train_handle, FLAGS.batches_per_eval_valid, train_writer,
                         prefix='TRAIN: ', is_training=True)
                report = eval(model, sess, FLAGS, valid_handle, FLAGS.batches_per_eval_valid, valid_writer,
                              prefix='VALIDATION: ')

                if report['acc'] > best_acc_validation:
                    best_acc_validation = report['acc']
                    model.saver_best.save(sess, FLAGS.path + "/cp_best.ckpt", write_meta_graph=False)

            # Train
            fetch = [model.train_op, model.global_step]
            feed = {model.is_training: True,
                    model.handle: train_handle}
            for i in tqdm(range(FLAGS.train_batches_per_epoch), desc='Train'):
                if i % 100 == 0:
                    _, step, summary = sess.run(fetch + [model.summary], feed_dict=feed)
                    train_writer.add_summary(summary, step)
                else:
                    _, step = sess.run(fetch, feed_dict=feed)

            # Checkpoint
            if (epochs_completed % 1 == 0) or (epochs_completed - 1 == FLAGS.num_epochs):
                model.saver.save(sess, FLAGS.path + "/cp.ckpt")  # global_step=model.global_step)

            epochs_completed += 1

        # Test set
        logging.info('FINISHED TRAINING, {} EPOCHS COMPLETED\n'.format(epochs_completed))
        eval(model, sess, FLAGS, test_handle, FLAGS.batches_per_eval_test, test_writer, prefix='TEST - LAST MODEL: ')
        Visual(sess, 'epoch_final', test_handle)

        model.saver.restore(sess, FLAGS.path + "/cp_best.ckpt")
        logging.info('Best validation accuracy: {:.3f}'.format(best_acc_validation))
        eval(model, sess, FLAGS, test_handle, FLAGS.batches_per_eval_test, test_writer, prefix='TEST - BEST MODEL: ')

        train_writer.close()
        valid_writer.close()
        test_writer.close()

def select_hyper_para_random(FLAGS):
    FLAGS.learning_rate_decay_factor = np.round(np.random.uniform(0.93, 0.99), 3)
    FLAGS.learning_rate_decay_steps = np.round(np.random.uniform(350, 650), 0)
    FLAGS.loc_std = np.round(np.random.uniform(0.075, 0.125), 3)
    FLAGS.learning_rate_RL = np.round(np.random.uniform(1., 1.4), 3)

    logging.info('CHOSEN PARAMS:\n'
                 'batch size:\t{}\n'
                 'MC samples:\t{}\n'
                 'loc std:\t{:.3f}\n'
                 'lrate decay\t{:.3f}\n'
                 'lrate decay step\t{}\n'
                 'lrate RL\t{:.3f}\n'.format(
                        FLAGS.batch_size, FLAGS.MC_samples, FLAGS.loc_std,
                        FLAGS.learning_rate_decay_factor, FLAGS.learning_rate_decay_steps,
                        FLAGS.learning_rate_RL))

    return FLAGS


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # Parsing experimental set up
    FLAGS, _ = Utility.parse_arg()
    # set img_shape, padding, num_classes according to dataset (ignoring cl inputs!)
    auto_adjust_flags(FLAGS)

    max_runs = 1
    if FLAGS.random_search:
        max_runs = 20

    for r in range(max_runs):
        if FLAGS.random_search:
            FLAGS = select_hyper_para_random(FLAGS)

        if not FLAGS.start_checkpoint:
            unknown_suffix = '_uk{}'.format(FLAGS.n_unknown_train) if FLAGS.open_set else ''
            experiment_name = '{}gl_bs{}_MC{}_std{}_dcay{}_step{}_lr{}_lrRL{}_{}sc{}_{}{}_Tanh{}_{}A'.format(
                                FLAGS.num_glimpses, FLAGS.batch_size, FLAGS.MC_samples, FLAGS.loc_std,
                                FLAGS.learning_rate_decay_factor, FLAGS.learning_rate_decay_steps,
                                FLAGS.learning_rate, FLAGS.learning_rate_RL, len(FLAGS.scale_sizes),
                                FLAGS.scale_sizes[0], FLAGS.cell, FLAGS.size_rnn_state, unknown_suffix,
                                FLAGS.exp_name_suffix)
        else:
            logging.info('CONTINUE TRAINING\n')
            experiment_name = FLAGS.start_checkpoint

        t_sz = (str(FLAGS.translated_size) if FLAGS.translated_size else "")
        FLAGS.path = FLAGS.summaries_dir + '/' + FLAGS.dataset + t_sz + '/' + experiment_name
        logging.info('\nPATH: ' + FLAGS.path + '\nCURRENT MODEL: ' + experiment_name + '\n')

        start_time = time.time()

        # load datasets
        train_data, valid_data, test_data = get_data(FLAGS)
        print("t - data loaded: {:.2f}s".format(time.time() - start_time))

        with tf.device('/device:GPU:*'):
            RAM = RAMNetwork(FLAGS=FLAGS,
                             full_summary=False)

        print("t - built graph: {:.2f}s".format(time.time() - start_time))

        train_model(RAM, FLAGS)

# tensorboard --logdir=logs/
# tensorboard --logdir=\\?\path for long paths due to windows max_len restriction