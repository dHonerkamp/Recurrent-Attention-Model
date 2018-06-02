import os
import argparse
import numpy as np


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
            default='/data/',
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
            default=500,
            help='Decay steps.')

        parser.add_argument(
            '--learning_rate_decay_factor',
            type=float,
            default=0.97,
            help='1 for no decay.')

        parser.add_argument(
            '--min_learning_rate',
            type=float,
            default=0.001,
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
            default=7,
            help='Number of Monte Carlo Samples per image.')

        parser.add_argument(
            '--num_epochs',
            type=int,
            default=200,
            help='Number of training epochs.')

        parser.add_argument(
            '--img_shape',
            type=list,
            default=[100,100,1],
            help='Image shape (including channels).'
                 'MNIST: [28,28,1] (even if translated)'
                 'MNIST_cluttered: [100,100,1]'
                 'cifar10: [32,32,3]')

        parser.add_argument(
            '--size_hidden_g',
            type=int,
            default=128,
            help='Hidden features in h_g')

        parser.add_argument(
            '--size_hidden_l',
            type=int,
            default=128,
            help='Hidden features in h_l')

        parser.add_argument(
            '--size_hidden_gl2',
            type=int,
            default=256,
            help='Hidden features in gl2 (second fc building on each h_l and h_g)')

        parser.add_argument(
            '--size_rnn_state',
            type=int,
            default=256,
            help='Dimensionality of the core networks RNN cell')

        parser.add_argument(
            '--loc_dim',
            type=int,
            default=2,
            help='Dimensionality of the locations (2 for x,y coordinates)')

        parser.add_argument(
            '--loc_std',
            type=float,
            default=0.15,
            help='Std used to sample locations. Relative to whole image being in range (-1, 1)')

        parser.add_argument(
            '--num_glimpses',
            type=int,
            default=4,
            help='Number of glimpses the network is allowed to take')

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
            default=1,
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
            default='MNIST_cluttered',
            help='What dataset to use. See main.get_data(). Atm:'
                 'MNIST, MNIST_cluttered, cifar10')

        parser.add_argument(
            '--translated_size',
            type=int,
            default=0,
            help='Size of the canvas to translate images on.')

        parser.add_argument(
            '--scale_sizes',
            type=list,
            default=[12, 24, 48, 96],
            help='List of scale dimensionalities used for retina network (size of the glimpses). '
                 'Resolution gets reduced to first glimpses size. '
                 'Should be ordered, smallest to largest scale.'
                 'Following scales must be a multiple of the first.'
                 'Might not work for uneven scale sizes!')

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



if __name__ == '__main__':
    FLAGS, unparsed = Utility.parse_arg()

    FLAGS_dict = vars(FLAGS)

    for k, v in FLAGS_dict.items():
        print('\t{} : {}'.format(k, v))
