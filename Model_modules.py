import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
from utility import weight_variable, bias_variable
# from tensorflow.python.framework import tensor_util

class RetinaSensor(object):
    def __init__(self, FLAGS):
        self.img_shape = FLAGS.img_shape
        self.scales = FLAGS.scale_sizes
        self.num_scales = tf.size(self.scales)
        self.padding = FLAGS.padding
        if FLAGS.resize_method == "AVG":
            self.resize_method = lambda glimpse, ratio: tf.nn.pool(glimpse,
                                                                 window_shape=[ratio, ratio],
                                                                 strides=[ratio, ratio],
                                                                 pooling_type="AVG",
                                                                 padding="SAME")
        elif FLAGS.resize_method == "BILINEAR":
            self.resize_method = lambda glimpse, ratio: tf.image.resize_images(glimpse,
                                               [self.scales[0], self.scales[0]],
                                               method=tf.image.ResizeMethod.BILINEAR)
        elif FLAGS.resize_method == "BICUBIC":
            self.resize_method = lambda glimpse, ratio: tf.image.resize_images(glimpse,
                                               [self.scales[0], self.scales[0]],
                                               method=tf.image.ResizeMethod.BICUBIC)

    def extract_glimpse_zero_padding_fix(self, img_batch, max_glimpse_width, max_glimpse_heigt, offset):

        with tf.name_scope('extract_glimpse_zero_padding_fix'):
            orig_sz = tf.constant(img_batch.get_shape().as_list()[1:3])
            padded_sz = orig_sz + tf.stack([max_glimpse_heigt, max_glimpse_width])

            img_batch_padded = tf.pad(img_batch,
                                      [(0, 0), (max_glimpse_heigt // 2, max_glimpse_heigt // 2),
                                       (max_glimpse_width // 2, max_glimpse_width // 2), (0, 0)])

            new_offset = offset * tf.cast(orig_sz, dtype=tf.float32) / tf.cast(padded_sz, tf.float32)

        return img_batch_padded, new_offset

    def __call__(self, img_NHWC, loc):
        '''
        Extract_glimpse pads the image with random noise!
        :param img_NHWC: [batch_size, height, width, channels]
        :param loc: [batch_size, 2]
        :return: [batch_sz, num_scales, H, W, C] and flattened for visualization: [batch_sz, num_scales*H*W*C]
        '''

        # img_NHWC_padded, adj_loc = extract_glimpse_zero_padding_fix(img_NHWC, self.scales[-1], self.scales[-1], loc)
        # ta = tf.TensorArray(dtype=tf.float32, size=self.num_scales, infer_shape=True, dynamic_size=False)
        # def _add_patch(i, ta):
        #    sc = tf.gather(self.scales, i)
        #    patch = tf.image.extract_glimpse(img_NHWC_padded, [sc, sc], adj_loc)
        #    patch = tf.image.resize_images(patch, [self.scales[0], self.scales[0]], method=tf.image.ResizeMethod.BILINEAR)  # BILINEAR would be faster but less accurate
        #    patch = flatten(patch)
        #    # patch = tf.reshape(patch, [tf.shape(patch)[0,1]])
        #    ta = ta.write(i, patch)
        #
        #    i += 1
        #    return i, ta
        #
        # final_i, final_ta = tf.while_loop(
        #    cond = lambda i, _: tf.less(i, self.num_scales),
        #    body = _add_patch,
        #    loop_vars = [tf.constant(0), ta]
        # )
        # patches = tf.transpose(final_ta.stack(), (1,0,2))  # out: [batch_sz, num_scales, pixel]
        #
        # d2 = tensor_util.constant_value(tf.pow(self.scales[0], 2) * self.img_size[-1])
        # d3 = tensor_util.constant_value(self.num_scales)
        # patches.set_shape([patches.shape[0], d3, d2])
        #
        # patches = flatten(patches)  # [batch_size, flat_image*num_scales]

        # Alternative to taking 3 glimpses
        if self.padding == "zero":
            img_NHWC_padded, adj_loc = self.extract_glimpse_zero_padding_fix(img_NHWC, self.scales[-1], self.scales[-1], loc)
            self.glimpse = tf.image.extract_glimpse(img_NHWC_padded, [self.scales[-1], self.scales[-1]], adj_loc)
        elif self.padding == "uniform":
            self.glimpse = tf.image.extract_glimpse(img_NHWC, [self.scales[-1], self.scales[-1]], loc, uniform_noise=True)
        elif self.padding == "normal":
            self.glimpse = tf.image.extract_glimpse(img_NHWC, [self.scales[-1], self.scales[-1]], loc, uniform_noise=False)

        # patches = []
        # for i, sc in enumerate(self.scales):
        #     if sc == self.scales[-1]:
        #         patch = glimpse
        #     else:
        #         start_end = (self.scales[-1] - sc) // 2
        #         patch = glimpse[:, start_end:-start_end, start_end:-start_end, :]
        #
        #     if sc is not self.scales[0]:
        #         ratio = sc//self.scales[0]
        #         patch = self.resize_method(patch, ratio)
        #
        #     patches.append(patch)
        #
        # patches = tf.stack(patches, axis=1)

        # tf.while should allow to process scales in parallel. Small improvement
        ta = tf.TensorArray(dtype=tf.float32, size=self.num_scales, infer_shape=True, dynamic_size=False)
        def _add_patch(i, ta):
            sc = tf.gather(self.scales, i)
            start_end = (self.scales[-1] - sc) // 2
            patch = self.glimpse[:, start_end:self.scales[-1] - start_end, start_end:self.scales[-1] - start_end, :]
            ratio = sc // self.scales[0]
            patch = self.resize_method(patch, ratio)

            ta = ta.write(i, patch)
            i += 1
            return i, ta

        final_i, final_ta = tf.while_loop(
               cond = lambda i, _: tf.less(i, self.num_scales),
               body = _add_patch,
               loop_vars = [tf.constant(0), ta],
            )


        # [batch_sz, num_scales, H, W, C]
        patches = tf.transpose(final_ta.stack(), [1,0,2,3,4])
        patches_flat = flatten(patches)
        return patches_flat


class GlimpseNetwork(object):
    def __init__(self, FLAGS, patch_shape):
        self.flat_patch_shape = np.prod(patch_shape)
        self.size_hidden_g = 128
        self.size_hidden_l = 128
        self.size_hidden_gl2 = 256

        # auxiliary_name_scope not supported by older tf version
        # with tf.variable_scope('fc_g', auxiliary_name_scope=False):
        with tf.variable_scope('fc_g') as s:
            with tf.name_scope(s.original_name_scope):
                self.g_W = weight_variable([self.flat_patch_shape, self.size_hidden_g], name='g_W')
                self.g_b = bias_variable([self.size_hidden_g], name='g_b')
        # with tf.variable_scope('fc_l', auxiliary_name_scope=False):
        with tf.variable_scope('fc_l') as s:
            with tf.name_scope(s.original_name_scope):
                self.l_W = weight_variable([FLAGS.loc_dim, self.size_hidden_l], name='l_W')
                self.l_b = bias_variable([self.size_hidden_l], name='l_b')
        # with tf.variable_scope('fc_g2', auxiliary_name_scope=False):
        with tf.variable_scope('fc_g2') as s:
            with tf.name_scope(s.original_name_scope):
                self.g2_W = weight_variable([self.size_hidden_g, self.size_hidden_gl2], name='g2_W')
                self.g2_b = bias_variable([self.size_hidden_gl2], name='g2_b')
        # with tf.variable_scope('fc_l2', auxiliary_name_scope=False):
        with tf.variable_scope('fc_l2') as s:
            with tf.name_scope(s.original_name_scope):
                self.l2_W = weight_variable([self.size_hidden_l, self.size_hidden_gl2], name='l2_W')
                self.l2_b = bias_variable([self.size_hidden_gl2], name='l2_b')

    def __call__(self, img_patch_flat, loc):
        with tf.variable_scope('fc_g'):
            h_g = tf.nn.relu(tf.matmul(img_patch_flat, self.g_W) + self.g_b)
        with tf.variable_scope('fc_l'):
            h_l = tf.nn.relu(tf.matmul(loc, self.l_W) + self.l_b)
        with tf.variable_scope('fc_g2'):
            h_g2 = tf.matmul(h_g, self.g2_W) + self.g2_b
        with tf.variable_scope('fc_l2'):
            h_l2 = tf.matmul(h_l, self.l2_W) + self.l2_b

        return tf.nn.relu(h_g2 + h_l2)


class GlimpseNetwork_DRAM(object):
    def __init__(self, FLAGS, patch_shape):
        self.patch_shape = patch_shape

        # DRAM configuration
        # self.size_hidden_conv1 = 64
        # self.size_hidden_conv2 = 64
        # self.size_hidden_conv3 = 128
        # self.size_hidden_fc_l  = 1024

        self.size_hidden_conv1 = 32
        self.size_hidden_conv2 = 32
        self.size_hidden_fc = 256

        with tf.variable_scope('g_conv'):
            self.conv1 = tf.layers.Conv2D(self.size_hidden_conv1,  [3, 3], padding='SAME', activation=tf.nn.relu, name='g_conv1')
            self.conv2 = tf.layers.Conv2D(self.size_hidden_conv2,  [3, 3], padding='SAME', activation=tf.nn.relu, name='g_conv2')
            # self.conv3 = tf.layers.Conv2D(self.size_hidden_conv3,  [3, 3], padding='SAME', activation=tf.nn.relu, name='g_conv3')
            self.fc_g  = tf.layers.Dense(self.size_hidden_fc, activation=tf.nn.relu, name='g_fc_g')
        with tf.variable_scope('fc_l'):
            self.fc_l = tf.layers.Dense(self.size_hidden_fc, activation=tf.nn.relu, name='g_fc_l')

    def __call__(self, img_patch_flat, loc):
        # bring into [batch_sz, num_scales *
        img_patch = tf.reshape(img_patch_flat, [-1] + self.patch_shape)
        img_patch = tf.unstack(img_patch, axis=1)
        patch = img_patch[0]
        for glimpse in img_patch[1:]:
            patch = tf.concat([patch, glimpse], axis=1)
        with tf.variable_scope('g_conv', reuse=tf.AUTO_REUSE):
            h_g1 = self.conv1(patch)
            h_g2 = self.conv2(h_g1)
            # h_g3 = self.conv3(h_g2)
            h_g4 = self.fc_g(flatten(h_g2))
        with tf.variable_scope('fc_l', reuse=tf.AUTO_REUSE):
            h_l = self.fc_l(loc)

        return h_g4 * h_l


class LocationNetwork(object):
    def __init__(self, FLAGS):
        # with tf.variable_scope('fc_locNet', auxiliary_name_scope=False):
        with tf.variable_scope('fc_locNet') as s:
            with tf.name_scope(s.original_name_scope):
                self.locNet_W = weight_variable([FLAGS.size_rnn_state, FLAGS.loc_dim], name='locNet_W')
                self.locNet_b = bias_variable([FLAGS.loc_dim], name='locNet_b')
        self.std = FLAGS.loc_std
        self.max_loc_rng = FLAGS.max_loc_rng

    def __call__(self, rnn_state, is_training):
        '''NOTE: tf backpropagates through sampling if using tf.distributions.Normal()'''
        with tf.variable_scope('fc_locNet'):
            loc_mean = tf.matmul(rnn_state, self.locNet_W) + self.locNet_b
            loc_mean = self.max_loc_rng * tf.nn.tanh(loc_mean)
            loc_mean = tf.clip_by_value(loc_mean, self.max_loc_rng *  -1, self.max_loc_rng * 1)

        with tf.variable_scope('sample_locs'):
            loc = tf.cond(is_training,
                   lambda: tf.distributions.Normal(loc=tf.stop_gradient(loc_mean), scale=self.std).sample(),
                   lambda: loc_mean)
        return loc, loc_mean


class LocationNetwork_inclLoc(object):
    def __init__(self, FLAGS):
        # with tf.variable_scope('fc_locNet', auxiliary_name_scope=False):
        with tf.variable_scope('fc_locNet') as s:
            with tf.name_scope(s.original_name_scope):
                self.locNet_W = weight_variable([FLAGS.size_rnn_state + FLAGS.loc_dim, FLAGS.loc_dim], name='locNet_W')
                self.locNet_b = bias_variable([FLAGS.loc_dim], name='locNet_b')
        self.std = FLAGS.loc_std
        self.max_loc_rng = FLAGS.max_loc_rng

    def __call__(self, rnn_state, loc, is_training):
        '''NOTE: tf backpropagates through sampling if using tf.distributions.Normal()'''
        with tf.variable_scope('fc_locNet'):
            state = tf.concat([rnn_state, loc], axis=1)
            loc_mean = tf.matmul(tf.stop_gradient(state), self.locNet_W) + self.locNet_b
            loc_mean = self.max_loc_rng * tf.nn.tanh(loc_mean)
            loc_mean = tf.clip_by_value(loc_mean, self.max_loc_rng *  -1, self.max_loc_rng * 1)

        with tf.variable_scope('sample_locs'):
            loc = tf.cond(is_training,
                   lambda: tf.random_normal(shape=loc_mean.shape, mean=loc_mean, stddev=self.std),  # tf.distributions.Normal(loc=loc_mean, scale=self.std).sample(),
                   lambda: loc_mean)
        return loc, loc_mean


class Rewards(object):
    def __init__(self, FLAGS):
        self.reward_type = "open_set" if FLAGS.open_set else "classification"
        self.num_glimpses = FLAGS.num_glimpses
        if FLAGS.open_set:
            self.unknown_label = FLAGS.unknown_label
        else:
            self.unknown_label = None

    def rewards_clf(self, predictions, label):
        reward = tf.cast(tf.equal(predictions, label), tf.float32)
        rewards = tf.concat([tf.zeros([tf.shape(predictions)[0], self.num_glimpses - 2], tf.float32),
                             tf.expand_dims(reward, axis=1)],
                            axis=1)  # [batch_sz, timesteps]

        return rewards, reward, tf.constant(0., tf.float32)

    def rewards_open_set(self, predictions, label):
        # penalize not identifying unknown as unknown with -1
        correct      = tf.equal(predictions, label)
        unknowns     = tf.equal(label, self.unknown_label)
        unknown_acc  = (tf.reduce_sum(tf.cast(tf.logical_and(unknowns, correct), tf.float32))
                        / tf.reduce_sum(tf.cast(unknowns, tf.float32)))
        penalty_mask = -1. * tf.cast(tf.logical_and( unknowns, tf.logical_not(correct) ), tf.float32)
        # either correct (correct=1, mask=0) or incorrect (correct=0, mask=0) or incorrect and unknown (correct=0, mask=-1)
        correct      = tf.cast(correct, tf.float32)
        reward       = correct + penalty_mask
        rewards      = tf.concat([tf.zeros([tf.shape(predictions)[0], self.num_glimpses - 2], tf.float32),
                                  tf.expand_dims(reward, axis=1)],
                                 axis=1)  # [batch_sz, timesteps]
        acc = tf.reduce_mean(correct)
        return rewards, reward, unknown_acc

    def __call__(self, predictions, label):
        if self.reward_type == "classification":
            rewards, reward, unknown_acc = self.rewards_clf(predictions, label)
        elif self.reward_type == "open_set":
            rewards, reward, unknown_acc = self.rewards_open_set(predictions, label)

        returns = tf.cumsum(rewards, axis=1, reverse=True)
        return returns, reward, unknown_acc

