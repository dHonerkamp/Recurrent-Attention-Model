import tensorflow as tf
import logging
from RAM import RAMNetwork
from input_fn import get_data
from utility import Utility, auto_adjust_flags, visualization
from tqdm import tqdm
import numpy as np
import time
import os





def eval(model, sess, FLAGS, handle, num_batches, writer, prefix, is_training=False):
    '''
    :param dataset: tuple of (init_op, data, placeholder)
    '''
    output_feed = [model.summary, model.global_step, model.accuracy, model.accuracy_MC, model.reward, model.loss, model.xent,
                   model.RL_loss, model.baselines_mse, model.learning_rate, model.unknown_accuracy]

    vars = ['acc', 'acc_MC', 'reward', 'loss', 'xent', 'elig', 'b_mse', 'lr', 'acc_unknown']
    averages = np.zeros(len(vars))

    for _ in tqdm(range(num_batches), desc='eval'):
        out = sess.run(output_feed, feed_dict={model.is_training: is_training,
                                               model.handle: handle})
        averages += np.array(out[2:]) / num_batches

    summs = [tf.Summary.Value(tag="batch_" + var, simple_value=avg) for var, avg in zip(vars, averages)]
    batch_values = tf.Summary(value=summs)
    step = out[1]
    writer.add_summary(batch_values, step)

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

        if FLAGS.start_checkpoint:
            model.saver.restore(sess, FLAGS.path + "/cp.ckpt")
            start_step = model.global_step.eval(session=sess)
            tf.logging.info('Training from step: %d ', start_step)
        else:
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(FLAGS.path + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.path + '/valid')
        test_writer  = tf.summary.FileWriter(FLAGS.path + '/test')

        train_handle = sess.run(model.train_init_op.string_handle())
        valid_handle = sess.run(model.valid_init_op.string_handle())
        test_handle  = sess.run(model.test_init_op.string_handle())
        sess.run(model.train_init_op.initializer, feed_dict={model.features_ph_train: train_data[0],
                                                             model.labels_ph_train: train_data[1]})
        sess.run(model.valid_init_op.initializer, feed_dict={model.features_ph_valid: valid_data[0],
                                                             model.labels_ph_valid: valid_data[1]})
        sess.run(model.test_init_op.initializer, feed_dict={model.features_ph_test: test_data[0],
                                                             model.labels_ph_test: test_data[1]})

        epochs_completed = 0
        best_acc_validation = 0.

        print("t - before train: {:.2f}s".format(time.time() - start_time))
        while epochs_completed < FLAGS.num_epochs:

            # Visualize
            if epochs_completed % 1 == 0:
                # visualize_glimpsePreds(sess, model, 'epoch_{}'.format(epochs_completed), valid_handle, FLAGS)
                visualization(sess, model, 'epoch_{}'.format(epochs_completed), valid_handle, FLAGS)

            # Train
            fetch = [model.train_op, model.global_step, model.reward, model.summary]
            for i in tqdm(range(FLAGS.train_batches_per_epoch), desc='train'):
                _, step, reward, summary = sess.run(fetch, feed_dict={model.is_training: True,
                                                                      model.handle: train_handle})
                if i % 100 == 0: train_writer.add_summary(summary, step)

            # Evaluate
            if epochs_completed % FLAGS.eval_step_interval == 0:
                _ = eval(model, sess, FLAGS, train_handle, FLAGS.batches_per_eval_valid, train_writer, prefix='TRAIN: ', is_training=True)
                report = eval(model, sess, FLAGS, valid_handle, FLAGS.batches_per_eval_valid, valid_writer, prefix='VALIDATION: ')

                if report['acc'] > best_acc_validation:
                    best_acc_validation = report['acc']
                    model.saver_best.save(sess, FLAGS.path + "/cp_best.ckpt", write_meta_graph=False)

            # Checkpoint
            if (epochs_completed % 1 == 0) or (epochs_completed - 1 == FLAGS.num_epochs):
                model.saver.save(sess, FLAGS.path + "/cp.ckpt")  # global_step=model.global_step)

            epochs_completed += 1

        # Test set
        logging.info('FINISHED TRAINING, {} EPOCHS COMPLETED\n'.format(epochs_completed))
        eval(model, sess, FLAGS, test_handle, FLAGS.batches_per_eval_test, test_writer, prefix='TEST - LAST MODEL: ')
        visualization(sess, model, 'epoch_final', test_handle, FLAGS)

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
            suffix = '_uk{}'.format(FLAGS.n_unknown_train) if FLAGS.open_set else ''
            experiment_name = '{}gl_bs{}_MC{}_std{}_dcay{}_step{}_lr{}_lrRL{}_{}sc{}_{}{}_Tanh'.format(
                                FLAGS.num_glimpses, FLAGS.batch_size, FLAGS.MC_samples, FLAGS.loc_std,
                                FLAGS.learning_rate_decay_factor, FLAGS.learning_rate_decay_steps,
                                FLAGS.learning_rate, FLAGS.learning_rate_RL, len(FLAGS.scale_sizes),
                                FLAGS.scale_sizes[0], FLAGS.cell, FLAGS.size_rnn_state) + suffix
        else:
            logging.info('CONTINUE TRAINING\n')
            experiment_name = FLAGS.start_checkpoint

        t_sz = (str(FLAGS.translated_size) if FLAGS.translated_size else "")
        FLAGS.path = FLAGS.summaries_dir + '/' + FLAGS.dataset + t_sz + '/' + experiment_name
        logging.info('\nPATH: ' + FLAGS.path + '\nCURRENT MODEL: ' + experiment_name + '\n')

        start_time = time.time()

        # load datasets
        train_data, valid_data, test_data = get_data(FLAGS)
        FLAGS.train_data_shape = (train_data[0].shape, train_data[1].shape)
        FLAGS.valid_data_shape = (valid_data[0].shape, valid_data[1].shape)
        FLAGS.test_data_shape  = (test_data[0].shape, test_data[1].shape)
        FLAGS.data_dtype = (train_data[0].dtype, train_data[1].dtype)
        print("t - data loaded: {:.2f}s".format(time.time() - start_time))

        with tf.device('/device:GPU:*'):
            RAM = RAMNetwork(FLAGS=FLAGS,
                             full_summary=False)

        print("t - built graph: {:.2f}s".format(time.time() - start_time))

        train_model(RAM, FLAGS)

# tensorboard --logdir=logs/
# tensorboard --logdir=\\?\path for long paths due to windows max_len restriction