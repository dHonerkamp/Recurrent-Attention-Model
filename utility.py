import argparse
import os
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def auto_adjust_flags(FLAGS):
    if FLAGS.dataset == "omniglot":
        FLAGS.img_shape = [105, 105, 1]
        FLAGS.padding = "zero"
        FLAGS.open_set = True  # wheter to punish not identifying a unknown image with reward -1
    if FLAGS.dataset == "MNIST_cluttered":
        FLAGS.img_shape = [100, 100, 1]
        FLAGS.padding = "zero"
        FLAGS.num_classes = 10
    if FLAGS.dataset == "MNIST":
        FLAGS.img_shape = [28, 28, 1]
        FLAGS.padding = "zero"
        FLAGS.num_classes = 10
    if FLAGS.dataset == "cifar10":
        FLAGS.img_shape = [32, 32, 3]
        FLAGS.padding = "uniform"
        FLAGS.num_classes = 10


class Utility(object):
    """
    A class for storing a number of useful functions
    """

    def __init__(self, debug=True):
        pass

    @staticmethod
    def parse_arg():
        """
        Parsing input arguments.

        Returns:
            Parsed data
        """
        parser = argparse.ArgumentParser(description='[*] RAM.')

        parser.add_argument(
            '--data_dir',
            type=str,
            default='data/',
            help="Where the data is stored")

        # parser.add_argument(
        #     '--model_path',
        #     type=str,
        #     default='../models',
        #     help='Path to the folder containing pre-trained model(s) [../models]')

        parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.001,
            help='How large a learning rate to use when training.')

        parser.add_argument(
            '--learning_rate_decay_steps',
            type=int,
            default=1000,
            help='Decay steps.')

        parser.add_argument(
            '--learning_rate_decay_factor',
            type=float,
            default=0.97,
            help='1 for no decay.')

        parser.add_argument(
            '--min_learning_rate',
            type=float,
            default=0.0001,
            help='Minimal learning rate.')

        parser.add_argument(
            '--learning_rate_RL',
            type=float,
            default=1,
            help='Relative weight of the RL objective.')

        parser.add_argument(
            '--batch_size',
            type=int,
            default=128,
            help='How many items to train with at once.')

        parser.add_argument(
            '--MC_samples',
            type=int,
            default=10,
            help='Number of Monte Carlo Samples per image.')

        parser.add_argument(
            '--num_epochs',
            type=int,
            default=35,
            help='Number of training epochs.')

        # parser.add_argument(
        #     '--img_shape',
        #     type=list,
        #     default=[100,100,1],
        #     help='Image shape (including channels).'
        #          'MNIST: [28,28,1] (even if translated)'
        #          'MNIST_cluttered: [100,100,1]'
        #          'cifar10: [32,32,3]')


        parser.add_argument(
            '--size_rnn_state',
            type=int,
            default=256,
            help='Dimensionality of the core networks RNN cell')

        parser.add_argument(
            '--cell',
            type=str,
            default='RNN',
            help='RNN cell to use.'
                 'RNN: cell from Mnih et al. 2014'
                 'LSTM: LSTM with relu activation')

        parser.add_argument(
            '--loc_dim',
            type=int,
            default=2,
            help='Dimensionality of the locations (2 for x,y coordinates)')

        parser.add_argument(
            '--loc_std',
            type=float,
            default=0.09,
            help='Std used to sample locations. Relative to whole image being in range (-1, 1)')

        parser.add_argument(
            '--num_glimpses',
            type=int,
            default=4,
            help='Number of glimpses the network is allowed to take')

        parser.add_argument(
            '--resize_method',
            type=str,
            default="BILINEAR",
            help='Method used to downsize the larger retina scales.'
                 'AVG: average pooling'
                 'BILINEAR'
                 'BICUBIC')

        parser.add_argument(
            '--max_gradient_norm',
            type=int,
            default=5,
            help='To clip gradients')

        parser.add_argument(
            '--summaries_dir',
            type=str,
            default='logs',
            help='Where to save summary logs for TensorBoard.')

        parser.add_argument(
            '--eval_step_interval',
            type=int,
            default=2,
            help='How often to evaluate the training results. In epochs.')

        parser.add_argument(
            '--random_search',
            type=bool,
            default=False,
            help='Run random search?')

        parser.add_argument(
            '--train_dir',
            type=str,
            default='../train',
            help='Directory to write event logs and checkpoint.')

        parser.add_argument(
            '--dataset',
            type=str,
            default='MNIST',
            help='What dataset to use. See main.get_data(). Atm:'
                 'MNIST, MNIST_cluttered, cifar10, omniglot')

        parser.add_argument(
            '--translated_size',
            type=int,
            default=0,
            help='Size of the canvas to translate images on.')

        parser.add_argument(
            '--scale_sizes',
            nargs='+',
            type=int,
            default=[8],
            help='List of scale dimensionalities used for retina network (size of the glimpses). '
                 'Resolution gets reduced to first glimpses size. '
                 'Should be ordered, smallest to largest scale.'
                 'Following scales must be a multiple of the first.'
                 'Might not work for uneven scale sizes!')

        parser.add_argument(
            '--max_loc_rng',
            type=float,
            default=1,
            help='In what range are the locations allowed to fall? (Max. is -1 to 1)')

        parser.add_argument(
            '--n_unknown_train',
            type=int,
            default=0,
            help='Currently only for omniglot. Number of alphabets used for unknown label during training')

        parser.add_argument(
            '--n_unknown_test',
            type=int,
            default=2,
            help='Currently only for omniglot. Completely held out alphabets. '
                 'Used as test: measure is ration of correctly classified as unknown')

        parser.add_argument(
            '--open_set',
            type=bool,
            default=False,
            help='Use unknown labels and punish not identifying them. Only omniglot atm.')

        parser.add_argument(
            '--start_checkpoint',
            type=str,
            default='',
            help='CLOSE TENSORBOARD! If specified, restore this pre-trained model before any training.')

        # parser.add_argument(
        #     '--save_step_interval',
        #     type=int,
        #     default=100,
        #     help='Save model checkpoint every save_steps.')
        #

        FLAGS, unparsed = parser.parse_known_args()

        return FLAGS, unparsed


def weight_variable(shape, name='w'):
    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=xavier_initializer())


def bias_variable(shape, name='b'):
    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=tf.zeros_initializer())



def plot_img_plus_locs(ax, batch_xs, batch_ys, preds, locs, im_shape, n, nr_examples, FLAGS):
    ax.imshow(batch_xs[n].reshape(im_shape))
    ax.set_title('Label: {} Prediction: {}'.format(batch_ys[n], preds[n]))

    for i in range(0, FLAGS.num_glimpses):
        c = ('green' if preds[n] == batch_ys[n]
             else 'red')

        if i == 0:
            marker = 'x'; fc = c
        else:
            marker = 'o'; fc = 'none'
        # plot glimpse location
        ax.scatter(locs[n, i, 1], locs[n, i, 0], marker=marker, facecolors=fc, edgecolors=c, linewidth=2.5,
                           s=0.25 * (5 * nr_examples * 24))
        # connecting line
        ax.plot(locs[n, i - 1:i + 1, 1], locs[n, i - 1:i + 1, 0], linewidth=2.5, color=c)
        # rectangle around location?
        # ax.add_patch(Rectangle(locs[n,i][::-1] - FLAGS.scale_sizes[0] / 2, width=FLAGS.scale_sizes[0], height=FLAGS.scale_sizes[0], edgecolor=c, facecolor='none'))

    ax.set_ylim([FLAGS.img_shape[0]-1, 0])
    ax.set_xlim([0, FLAGS.img_shape[1]-1])
    ax.set_xticks([])
    ax.set_yticks([])


def plot_composed_glimpse(ax, gl_composed, step_preds, n, i, FLAGS):
    ax.imshow(gl_composed[i][n].reshape(2 * [np.max(FLAGS.scale_sizes)] + [FLAGS.img_shape[-1]]).squeeze())
    ax.set_title('class {}: {:.3f}'.format(step_preds[i][0][n], step_preds[i][1][n]))
    ax.set_xticks([])
    ax.set_yticks([])


def visualization(sess, model, prefix, handle, FLAGS, nr_examples=8):
    '''
    Plot nr_examples images together with the extracted locations and glimpses
    :param dataset: tuple of (init_op, data, placeholder)
    '''
    # atm running for same batch_sz as training,because extract_glimpse cannot cope with varying batch_sz
    os.makedirs(os.path.join(FLAGS.path, 'glimpses'), exist_ok=True)

    output_feed = [model.global_step, model.x, model.y, model.locs,
                   model.glimpses_composed, model.prediction, model.intermed_preds]
    step, batch_xs, batch_ys, locs, gl_composed, preds, step_preds = sess.run(output_feed, feed_dict={model.is_training: False,
                                                                                                      model.handle: handle})
    # extract_glimpses: (-1,-1) is top left. 0 is y-axis, 1 is x-axis. Scale of imshow shifted by 1.
    locs = np.clip(locs, -1, 1)
    locs =  (locs/2 + 0.5) * FLAGS.img_shape[1::-1] - 1  # in img_shape y comes first, then x

    f, axes = plt.subplots(nr_examples, FLAGS.num_glimpses + 1, figsize=(6*FLAGS.num_glimpses, 5*nr_examples))
    axes = axes.reshape([nr_examples, FLAGS.num_glimpses+1])

    im_shape = (FLAGS.img_shape[:2] if FLAGS.img_shape[2] == 1
                else FLAGS.img_shape)

    for n in range(nr_examples):
        plot_img_plus_locs(axes[n, 0], batch_xs, batch_ys, preds, locs, im_shape, n, nr_examples, FLAGS)

        for i in range(0, FLAGS.num_glimpses):
            plot_composed_glimpse(axes[n, i + 1], gl_composed, step_preds, n, i, FLAGS)


    f.tight_layout()
    f.savefig('{}/glimpses/{}_{}.png'.format(FLAGS.path, step, prefix), bbox_inches='tight')
    plt.close(f)



if __name__ == '__main__':
    FLAGS, unparsed = Utility.parse_arg()

    FLAGS_dict = vars(FLAGS)

    for k, v in FLAGS_dict.items():
        print('\t{} : {}'.format(k, v))
