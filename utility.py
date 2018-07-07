import argparse
import os
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def auto_adjust_flags(FLAGS):
    if FLAGS.dataset == "omniglot":
        FLAGS.img_shape        = [105, 105, 1]
        FLAGS.padding          = "zero"
        FLAGS.open_set         = True  # whether to punish not identifying a unknown image with reward -1
        FLAGS.ConvGlimpse      = False
    if FLAGS.dataset == "MNIST_cluttered":
        FLAGS.img_shape        = [100, 100, 1]
        FLAGS.padding          = "zero"
        FLAGS.num_classes      = 10
        FLAGS.ConvGlimpse      = False
    if FLAGS.dataset == "MNIST":
        FLAGS.img_shape        = [28, 28, 1]
        FLAGS.padding          = "zero"
        FLAGS.num_classes      = 10
        FLAGS.ConvGlimpse      = False
    if FLAGS.dataset == "cifar10":
        FLAGS.img_shape        = [32, 32, 3]
        FLAGS.padding          = "uniform"
        FLAGS.num_classes      = 10
        FLAGS.ConvGlimpse      = True
        FLAGS.size_glimpse_out = 1024


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
            '--exp_name_suffix',
            type=str,
            default='A',
            help="Experiment name suffix. Used for log folder.")

        parser.add_argument(
            '--data_dir',
            type=str,
            default='data/',
            help="Where the data is stored")

        parser.add_argument(
            '--num_parallel_preprocess',
            type=int,
            default=4,
            help="Parallel processes during pre-processing (on CPU)")

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
            '--size_glimpse_out',
            type=int,
            default=256,
            help='Dimensionality of the lsat layer of the glimpse network.')

        parser.add_argument(
            '--size_rnn_state',
            type=int,
            default=256,
            help='Dimensionality of the core networks RNN cell')

        parser.add_argument(
            '--cell',
            type=str,
            default='LSTM',
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

if __name__ == '__main__':
    FLAGS, unparsed = Utility.parse_arg()

    FLAGS_dict = vars(FLAGS)

    for k, v in FLAGS_dict.items():
        print('\t{} : {}'.format(k, v))
