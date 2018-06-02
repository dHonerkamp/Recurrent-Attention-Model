import tensorflow as tf
import logging
from model import RAMNetwork
from utility import Utility
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import time


def store_visualization(sess, model, prefix, valid_handle, FLAGS, nr_examples=8):
    '''
    Plot nr_examples images together with the extracted locations and glimpses
    :param sess:
    :param model:
    :param nr_examples:
    :param prefix:
    :param FLAGS:
    :return:
    '''
    # atm running for same batch_sz as training,because extract_glimpse cannot cope with varying batch_sz
    # sess.run(model.validation_init_op)

    output_feed = [model.global_step, model.x, model.y, model.locs, model.glimpses_composed, model.prediction]
    step, batch_xs, batch_ys, locs, glimpses, preds = sess.run(output_feed, feed_dict={model.is_training: False,
                                                                                       model.handle: valid_handle})
    # extract_glimpses: (-1,-1) is top left. 0 is y-axis, 1 is x-axis. Scale of imshow shifted by 1.
    locs = np.clip(locs, -1, 1)
    locs =  (locs/2 + 0.5) * FLAGS.img_shape[1::-1] - 1  # in img_shape y comes first, then x

    f, axes = plt.subplots(nr_examples, FLAGS.num_glimpses + 1, figsize=(6*FLAGS.num_glimpses, 5*nr_examples))
    axes = axes.reshape([nr_examples, FLAGS.num_glimpses+1])

    if FLAGS.img_shape[2] == 1: im_shape = FLAGS.img_shape[:2]
    else: im_shape = FLAGS.img_shape

    for n in range(nr_examples):
        axes[n, 0].imshow(batch_xs[n].reshape(im_shape))
        axes[n, 0].set_title('Label: {} Prediction: {}'.format(np.argmax(batch_ys[n]), preds[n]))

        for i in range(0, FLAGS.num_glimpses):
            if preds[n] == np.argmax(batch_ys[n]): c = 'green'
            else: c = 'red'

            if i == 0: marker='x'; fc=c
            else: marker='o'; fc='none'
            # plot glimpse location
            axes[n, 0].scatter(locs[n, i, 1], locs[n, i, 0], marker=marker, facecolors=fc, edgecolors=c, linewidth=2.5, s=0.25*(5*nr_examples*24))
            # connecting line
            axes[n, 0].plot(locs[n, i - 1:i + 1, 1], locs[n, i - 1:i + 1, 0], linewidth=2.5, color=c)
            # rectangle around location?
            # axes[n, 0].add_patch(Rectangle(locs[n,i][::-1] - FLAGS.scale_sizes[0] / 2, width=FLAGS.scale_sizes[0], height=FLAGS.scale_sizes[0], edgecolor=c, facecolor='none'))
            # plot glimpse
            axes[n, i + 1].imshow(glimpses[i][n].reshape(2*[np.max(FLAGS.scale_sizes)] + [FLAGS.img_shape[-1]]).squeeze())
            axes[n, i + 1].set_xticks([])
            axes[n, i + 1].set_yticks([])

        axes[n, 0].set_ylim([FLAGS.img_shape[0]-1, 0])
        axes[n, 0].set_xlim([0, FLAGS.img_shape[1]-1])
        axes[n, 0].set_xticks([])
        axes[n, 0].set_yticks([])

    f.tight_layout()
    f.savefig('{}/train/{}_{}.png'.format(FLAGS.path, step, prefix), bbox_inches='tight')

    plt.close(f)


def eval(model, sess, init_handle, num_batches, writer, prefix, is_training=False):
    # sess.run(init_op)
    output_feed = [model.summary, model.global_step, model.accuracy, model.loss, model.xent,
                   model.eligibility, model.baselines_mse, model.learning_rate]
    if is_training:
        output_feed.append(model.train_op)

    vars = ['acc', 'loss', 'xent', 'elig', 'b_mse', 'lr']
    averages = len(output_feed) * [0]

    for ii in range(num_batches):
        out = sess.run(output_feed, feed_dict={model.is_training: is_training,
                                               model.handle: init_handle})
        step = out[1]
        if ii == 0:
            writer.add_summary(out[0], step)

        for v in range(len(vars)):
            averages[v] += out[v+2] / num_batches

    batch_values = tf.Summary(value=[
        tf.Summary.Value(tag="batch_acc", simple_value=averages[0]),
        tf.Summary.Value(tag="batch_loss", simple_value=averages[1]),
        tf.Summary.Value(tag="batch_xent", simple_value=averages[2]),
        tf.Summary.Value(tag="batch_elig", simple_value=averages[3]),
        tf.Summary.Value(tag="batch_b_mse", simple_value=averages[4]),
    ])
    # writer.add_summary(out[0], step)
    writer.add_summary(batch_values, step)

    strs = [item for pair in zip(vars, averages) for item in pair]
    s = 'step ' + str(step) + ' - ' + prefix + len(vars) * ' {}: {:.3f}'
    logging.info(s.format(*strs))

    report = dict(zip(vars, averages))
    report['step'] = step
    report['MC_sampling'] = FLAGS.MC_samples
    report['loc_std'] = FLAGS.loc_std

    return report


def empty_results():
    return {'acc': 0,
            'loss': 0,
            'xent': 0,
            'elig': 0,
            'b_mse': 0,
            'lr': 0,
            'step': 0,
            'loc_std': 0}


def train_model(model, FLAGS):

    with tf.Session() as sess:

        if FLAGS.start_checkpoint:
            model.saver.restore(sess, FLAGS.path + "/cp.ckpt")
            start_step = model.global_step.eval(session=sess)
            tf.logging.info('Training from step: %d ', start_step)
        else:
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(FLAGS.path + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.path + '/valid', sess.graph)
        test_writer  = tf.summary.FileWriter(FLAGS.path + '/test', sess.graph)

        train_handle = sess.run(model.train_init_op.string_handle())
        valid_handle = sess.run(model.valid_init_op.string_handle())


        epochs_completed = 0
        best_acc_validation = 0.


        print("t - before train: {:.2f}s".format(time.time() - start_time))
        while epochs_completed < FLAGS.num_epochs:

            # Visualize
            if epochs_completed % 6 == 0:
                store_visualization(sess, model, 'epoch_{}'.format(epochs_completed), valid_handle, FLAGS)
            if epochs_completed <=1: print("t - visual: {:.2f}s".format(time.time() - start_time))

            # Train
            for _ in range(FLAGS.train_batches_per_epoch):
                sess.run([model.train_op], feed_dict={model.is_training: True,
                                                      model.handle: train_handle})
            if epochs_completed <= 1: print("t - epoch: {:.2f}s".format(time.time() - start_time))

            # Evaluate
            if epochs_completed % FLAGS.eval_step_interval == 0:
                _ = eval(model, sess, train_handle, FLAGS.train_batches_per_epoch, train_writer, prefix='TRAIN: ', is_training=True)
                report = eval(model, sess, valid_handle, FLAGS.batches_per_eval_valid, valid_writer, prefix='VALIDATION: ')

                if report['acc'] > best_acc_validation:
                    best_acc_validation = report['acc']
                    model.saver_best.save(sess, FLAGS.path + "/cp_best.ckpt")

            # Checkpoint
            if (epochs_completed % 5 == 0) or (epochs_completed - 1 == FLAGS.num_epochs):
                model.saver.save(sess, FLAGS.path + "/cp.ckpt")  # global_step=model.global_step)

            epochs_completed += 1

        # Test set
        logging.info('FINISHED TRAINING, {} EPOCHS COMPLETED\n'.format(epochs_completed))
        test_handle = sess.run(model.test_init_op.string_handle())
        eval(model, sess, test_handle, FLAGS.batches_per_eval_test, test_writer, prefix='TEST - LAST MODEL: ')
        store_visualization(sess, model, 'epoch_final', valid_handle, FLAGS)

        model.saver.restore(sess, FLAGS.path + "/cp_best.ckpt")
        logging.info('Best validation accuracy: {:.3f}'.format(best_acc_validation))
        eval(model, sess, test_handle, FLAGS.batches_per_eval_test, test_writer, prefix='TEST - BEST MODEL: ')


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

    max_runs = 1
    if FLAGS.random_search:
        max_runs = 20

    for r in range(max_runs):
        if FLAGS.random_search:
            FLAGS = select_hyper_para_random(FLAGS)

        if not FLAGS.start_checkpoint:
            experiment_name = '{}glimpses_bs{}_MC{}_std{}_decay{}_step{}_lr{}_lrRL{}_minLR{}_{}sc{}'.format(
                                FLAGS.num_glimpses, FLAGS.batch_size, FLAGS.MC_samples, FLAGS.loc_std,
                                FLAGS.learning_rate_decay_factor, FLAGS.learning_rate_decay_steps,
                                FLAGS.learning_rate, FLAGS.learning_rate_RL, FLAGS.min_learning_rate,
                                len(FLAGS.scale_sizes), FLAGS.scale_sizes[0])
        else:
            logging.info('CONTINUE TRAINING\n')
            experiment_name = FLAGS.start_checkpoint

        if FLAGS.translated_size:
            t_sz= str(FLAGS.translated_size)
        else:
            t_sz = ""
        FLAGS.path = FLAGS.summaries_dir + '/' + FLAGS.dataset + t_sz + '/' + experiment_name
        logging.info('\nPATH: ' + FLAGS.path + '\nCURRENT MODEL: ' + experiment_name + '\n')

        start_time = time.time()

        with tf.device('/device:GPU:*'):
            RAM = RAMNetwork(FLAGS = FLAGS,
                             patch_shape=len(FLAGS.scale_sizes) * np.power(FLAGS.scale_sizes[0], 2) * FLAGS.img_shape[-1],
                             num_glimpses = FLAGS.num_glimpses,
                             batch_size = FLAGS.batch_size * FLAGS.MC_samples,
                             full_summary=False)

        print("t - build graph: {:.2f}s".format(time.time() - start_time))

        train_model(RAM, FLAGS)

# tensorboard --logdir=logs/