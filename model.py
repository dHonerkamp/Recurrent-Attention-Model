import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, flatten
from tensorflow.python.framework import tensor_util
from input_fn import input_fn


def weight_variable(shape, name='w'):
    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=xavier_initializer())


def bias_variable(shape, name='b'):
    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=tf.zeros_initializer())


def extract_glimpse_zero_padding_fix(img_batch, max_glimpse_width, max_glimpse_heigt, offset):
    '''
    Source: https://github.com/tensorflow/tensorflow/issues/2134#issuecomment-326071624
    Taking a closer look, seems really unneccessary, just pad it instead, much faster.
    Uncommented stuff can be deleted if no issues arise.
    '''
    with tf.name_scope('extract_glimpse_zero_padding_fix'):
        orig_sz = tf.constant(img_batch.get_shape().as_list()[1:3])
        # not sure why still random noise if dividing by 2. Seems like a documented bug in extract_glimpse
        padded_sz = orig_sz + tf.stack([max_glimpse_heigt, max_glimpse_width])

        # batch_sz = offset.shape[0]
        #
        # # unstack images in batch
        # img_batch_unstacked = tf.unstack(img_batch, num=batch_sz, axis=0)
        #
        # # stack images on channels
        # concat_on_channel_batch = tf.concat(img_batch_unstacked, axis=2)

        # # pad the image with max glimpse width/height
        # resized_img_batch = tf.image.resize_image_with_crop_or_pad(
        #     image=concat_on_channel_batch,
        #     target_width=tf.cast(padded_sz[1], tf.int32),
        #     target_height=tf.cast(padded_sz[0], tf.int32)
        # )
        #
        # # undo the operations to get the original batch
        # # first split images on channels
        # splited_on_channel_batch = tf.split(
        #     resized_img_batch,
        #     num_or_size_splits=batch_sz,
        #     axis=2
        # )
        #
        # # combine the images back to the original shape
        # img_batch_padded = tf.stack(
        #     splited_on_channel_batch,
        #     axis=0
        # )

        img_batch_padded = tf.pad(img_batch,
                                  [(0, 0), (max_glimpse_heigt // 2, max_glimpse_heigt // 2),
                                              (max_glimpse_width // 2, max_glimpse_width // 2), (0, 0)])

        new_offset = offset * tf.cast(orig_sz, dtype=tf.float32) / tf.cast(padded_sz, tf.float32)

    return img_batch_padded, new_offset


class RetinaSensor(object):
    def __init__(self, img_size, scale_sizes):
        '''
        :param img_size:
        :param scale_sizes: list of scales
        '''
        self.img_size = img_size
        self.scales = scale_sizes
        self.num_scales = tf.size(self.scales)

    def __call__(self, img_NHWC, loc):
        '''
        Extract_glimpse pads the image with random noise!
        :param img_NHWC: [batch_size, height, width, channels]
        :param loc: [batch_size, 2]
        :return: [batch_size, flat_image*num_scales]
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
        batch_sz, _, _, channel = img_NHWC.shape
        img_NHWC_padded, adj_loc = extract_glimpse_zero_padding_fix(img_NHWC, self.scales[-1], self.scales[-1], loc)
        glimpse = tf.image.extract_glimpse(img_NHWC_padded, [self.scales[-1], self.scales[-1]], adj_loc)
        patches = tf.image.resize_images(glimpse,
                                           [self.scales[0], self.scales[0]],
                                           method=tf.image.ResizeMethod.BILINEAR)
        patches = flatten(patches)
        for sc in self.scales[:-1]:
            start_end = (self.scales[-1] - sc) // 2
            patch = glimpse[:, start_end:-start_end, start_end:-start_end, :]
            patch = tf.image.resize_images(patch,
                                           [self.scales[0], self.scales[0]],
                                           method=tf.image.ResizeMethod.BILINEAR)
            patches = tf.concat([patches, flatten(patch)], axis=1)

        return patches


class GlimpseNetwork(object):
    def __init__(self, hidden_g, hidden_l, hidden_gl2, img_size, scale_sizes, patch_shape, loc_dim):
        self.sensor = RetinaSensor(img_size, scale_sizes)

        with tf.variable_scope('fc_g', auxiliary_name_scope=False):
            self.g_W = weight_variable([patch_shape, hidden_g], name='g_W')
            self.g_b = bias_variable([hidden_g], name='g_b')
        with tf.variable_scope('fc_l', auxiliary_name_scope=False):
            self.l_W = weight_variable([loc_dim, hidden_l], name='l_W')
            self.l_b = bias_variable([hidden_l], name='l_b')
        with tf.variable_scope('fc_g2', auxiliary_name_scope=False):
            self.g2_W = weight_variable([hidden_g, hidden_gl2], name='g2_W')
            self.g2_b = bias_variable([hidden_gl2], name='g2_b')
        with tf.variable_scope('fc_l2', auxiliary_name_scope=False):
            self.l2_W = weight_variable([hidden_l, hidden_gl2], name='l2_W')
            self.l2_b = bias_variable([hidden_gl2], name='l2_b')

    def __call__(self, img_NHWC, loc):
        with tf.variable_scope('retina_sensor'):
            img_patch = self.sensor(img_NHWC, tf.clip_by_value(loc, -1, 1))
        with tf.variable_scope('fc_g'):
            h_g = tf.nn.relu(tf.matmul(img_patch, self.g_W) + self.g_b)
        with tf.variable_scope('fc_l'):
            h_l = tf.nn.relu(tf.matmul(loc, self.l_W) + self.l_b)
        with tf.variable_scope('fc_g2'):
            h_g2 = tf.matmul(h_g, self.g2_W) + self.g2_b
        with tf.variable_scope('fc_l2'):
            h_l2 = tf.matmul(h_l, self.l2_W) + self.l2_b

        return tf.nn.relu(h_g2 + h_l2), img_patch


class LocationNetwork(object):
    def __init__(self, loc_dim, loc_std, size_rnn_state):
        with tf.variable_scope('fc_locNet', auxiliary_name_scope=False):
            self.locNet_W = weight_variable([size_rnn_state, loc_dim], name='locNet_W')
            self.locNet_b = bias_variable([loc_dim], name='locNet_b')
        self.std = loc_std

    def __call__(self, rnn_state, is_training):
        '''NOTE: tf backpropagates through sampling'''
        with tf.variable_scope('fc_locNet'):
            loc_mean = tf.matmul(tf.stop_gradient(rnn_state), self.locNet_W) + self.locNet_b
            # loc_mean = tf.nn.tanh(loc_mean)

        with tf.variable_scope('sample_locs'):
            loc = tf.cond(is_training,
                   lambda: tf.distributions.Normal(loc=loc_mean, scale=self.std).sample(),
                   lambda: loc_mean)
        return loc, loc_mean


class _rnn_cell_RAM(tf.nn.rnn_cell.RNNCell):
    '''Adjusted BasicRNNCell processing h(t-1) and x(t) separately,
    then adding them instead of concatenating.'''
    def __init__(self, num_units, activation=None, reuse=None, name=None):
        super().__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = tf.python.layers.base.InputSpec(ndim=2)

        self._num_units = num_units
        self._activation = activation or tf.nn.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        # self._kernel_input = self.add_variable(
        #     "kernel_input",
        #     shape=[input_depth, self._num_units])
        # self._bias_input = self.add_variable(
        #     'bias_input',
        #     shape=[self._num_units],
        #     initializer=tf.zeros_initializer(dtype=self.dtype))
        #
        # self._kernel_hidden = self.add_variable(
        #     "kernel_hidden",
        #     shape=[self._num_units, self._num_units])
        # self._bias_hidden = self.add_variable(
        #     'bias_hidden',
        #     shape=[self._num_units],
        #     initializer=tf.zeros_initializer(dtype=self.dtype))

        self._kernel = self.add_variable(
            'kernel',
            shape=[input_depth + self._num_units, self._num_units])
        self._bias = self.add_variable(
            'bias',
            shape=[self._num_units],
            initializer=tf.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        # gate_inputs = tf.matmul(inputs, self._kernel_input)
        # gate_inputs = tf.nn.bias_add(gate_inputs, self._bias_input)
        # gate_hidden = tf.matmul(state, self._kernel_hidden)
        # gate_hidden = tf.nn.bias_add(gate_hidden, self._bias_hidden)
        # output = self._activation(gate_inputs + gate_hidden)

        gate = tf.matmul(tf.concat([inputs, state], 1), self._kernel)
        gate = tf.nn.bias_add(gate, self._bias)
        output = self._activation(gate)

        return output, output


class RAMNetwork():
    def __init__(self, FLAGS, patch_shape, num_glimpses, batch_size, full_summary=False,):
        '''Input:
        img_shape: [H,W,C]
        '''
        tf.reset_default_graph()

        with tf.name_scope('Placeholders'):
            self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        with tf.device('/device:CPU:*'):
            with tf.name_scope('Dataset'):
                inputs = input_fn(FLAGS)
                self.handle        = inputs['handle']
                self.train_init_op = inputs['train_init_op']
                self.valid_init_op = inputs['valid_init_op']
                self.test_init_op  = inputs['test_init_op']

                # TODO: OR FOR MC JUST SAMPLE 10 DIFF. LOCATIONS AROUND SAME MEAN AND AVERAGE ...
                # PREDICTIONS? ATM FOLLOWING GLIMPSES DIFFER COMPLETELY AS CELL STATES DIFFER
                (self.x, self.y) = tf.cond(self.is_training,
                                           lambda: (tf.tile(inputs['images'], [FLAGS.MC_samples, 1, 1, 1]),
                                                    tf.tile(inputs['labels'], [FLAGS.MC_samples, 1])),
                                           lambda: (inputs['images'],
                                                    inputs['labels'])
                                           )

                batch_sz = tf.shape(self.x)[0]  # potentially use variable batch_size
                num_classes = self.y.shape[1]

                img_NHWC = tf.reshape(self.x, [batch_sz] + FLAGS.img_shape)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.maximum(
                                    tf.train.exponential_decay(
                                            FLAGS.learning_rate,
                                            self.global_step,
                                            FLAGS.learning_rate_decay_steps,
                                            FLAGS.learning_rate_decay_factor,
                                            staircase=True),
                                    FLAGS.min_learning_rate)

        with tf.variable_scope('LocationNetwork', reuse=tf.AUTO_REUSE):
            location_network = LocationNetwork(FLAGS.loc_dim, FLAGS.loc_std, FLAGS.size_rnn_state)
            locs_ta      = tf.TensorArray(tf.float32, size=num_glimpses, name='locs_ta')
            loc_means_ta = tf.TensorArray(tf.float32, size=num_glimpses, name='loc_means_ta')

        with tf.variable_scope('GlimpseNetwork', reuse=tf.AUTO_REUSE):
            glimpse_network = GlimpseNetwork(FLAGS.size_hidden_g,
                                             FLAGS.size_hidden_l,
                                             FLAGS.size_hidden_gl2,
                                             FLAGS.img_shape,
                                             FLAGS.scale_sizes,
                                             patch_shape,
                                             FLAGS.loc_dim)
            # keep track of glimpses to visualize
            glimpses_ta = tf.TensorArray(tf.float32, size=num_glimpses, name='glimpses_ta')

        with tf.name_scope('CoreNetwork'):
            cell = _rnn_cell_RAM(FLAGS.size_rnn_state, activation=tf.nn.relu)
            output_ta = (locs_ta, loc_means_ta, glimpses_ta)

            def loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output

                if cell_output is None:  # time == 0
                    with tf.variable_scope('GlimpseNetwork', reuse=True):
                        loc = tf.random_uniform((batch_size, FLAGS.loc_dim), minval=-1, maxval=1)
                        loc_mean = loc

                    with tf.variable_scope('GlimpseNetwork', reuse=True):
                        glimpse, img_patch = glimpse_network(img_NHWC, loc)

                    next_cell_state = cell.zero_state(batch_sz, tf.float32)
                    loop_state = output_ta

                else:  # time == 1+
                    next_cell_state = cell_state

                    with tf.variable_scope('LocationNetwork', reuse=tf.AUTO_REUSE):
                        loc, loc_mean = location_network(cell_state, self.is_training)

                    with tf.variable_scope('GlimpseNetwork', reuse=tf.AUTO_REUSE):
                        # tf automatically reparametrizes the normal dist., but we don't want to propagate the supervised loss into location
                        glimpse, img_patch = glimpse_network(img_NHWC, tf.stop_gradient(loc))

                with tf.name_scope('write_or_finished'):
                    elements_finished = (time >= num_glimpses)
                    finished = tf.reduce_all(elements_finished)

                    def _write():
                        return (loop_state[0].write(time, loc),
                                loop_state[1].write(time, loc_mean),
                                loop_state[2].write(time, img_patch))
                    next_loop_state = tf.cond(
                        finished,
                        lambda: loop_state,
                        lambda: _write())

                return (elements_finished, glimpse, next_cell_state,
                        emit_output, next_loop_state)

            outputs_ta, final_state, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
            rnn_outputs = outputs_ta.stack(name='stack_rnn_outputs')  # [time, batch_sz, num_cell]

        with tf.name_scope('stack_locs'):
            self.locs = tf.transpose(loop_state_ta[0].stack(name='stack_locs'), [1,0,2])  # [batch_sz, timesteps, loc_dims]
            loc_means = tf.transpose(loop_state_ta[1].stack(name='stack_loc_means'), [1,0,2])
            self.glimpses = loop_state_ta[2].stack(name='stack_glimpses')

        # Training baseline_t. Implementation taken from tflearn.time_distributed
        with tf.variable_scope('Baseline'):
            fc_b = tf.layers.Dense(1, activation=None, kernel_initializer=xavier_initializer())
            # [batch_size, size_rnn_state] x [size_rnn_state, 1] = [batch_sz, 1] for t in range(num_glimpses)
            baselines = [tf.squeeze(fc_b(tf.stop_gradient(rnn_outputs[i]))) for i in range(num_glimpses)]
            baselines = tf.stack(baselines, axis=1)  # [batch_sz, timesteps]

        # classification after last time-step
        with tf.variable_scope('CoreNetwork_preds'):
            logits = tf.layers.dense(rnn_outputs[-1], num_classes, kernel_initializer=xavier_initializer(), name='fc_logits')
            preds = tf.nn.softmax(logits)
            self.prediction = tf.argmax(logits, 1)

        # classification loss
        with tf.name_scope('Cross-entropy_loss'):
            self.xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits))

        # agent rewards
        with tf.name_scope('Rewards'):
            with tf.name_scope('reward'):
                reward = tf.cast(tf.equal(self.prediction, tf.argmax(self.y, 1)), tf.float32)
                rewards = tf.concat([tf.zeros([batch_sz, num_glimpses-1], tf.float32), tf.expand_dims(reward, axis=1)],  axis=1)  # [batch_sz, timesteps]
                self.returns = tf.cumsum(rewards, axis=1, reverse=True)

            with tf.name_scope('advantages'):
                # stop_gradient because o/w error from eligibility propagated into baselines.
                # But we want to train them independently by MSE
                self.advantages = self.returns - baselines

            with tf.name_scope('loglikelihood'):
                # only want gradients flow through the suggested mean
                z = (tf.stop_gradient(self.locs) - loc_means) / FLAGS.loc_std  # [batch_sz, timesteps, loc_dims]
                loglik = -0.5 * tf.reduce_sum(tf.square(z), axis=2)

            with tf.name_scope('eligibility'):
                # do not propagate back through logits?
                self.eligibility = tf.reduce_mean(loglik * tf.stop_gradient(self.advantages))

        # baseline loss
        with tf.name_scope('Baseline_loss'):
            self.baselines_mse = tf.reduce_mean(tf.square(tf.stop_gradient(self.returns) - baselines))

        # hybrid loss
        with tf.name_scope('Hybrid_loss'):
            self.loss = - FLAGS.learning_rate_RL * self.eligibility + self.xent + self.baselines_mse

        with tf.variable_scope('Adam'):
            train_op = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = train_op.compute_gradients(self.loss)

            # look at selected gradients
            self.grads = tf.gradients(self.loss, [loc_means, location_network.locNet_W, location_network.locNet_b])

            clipped_grads_and_vars = [(tf.clip_by_norm(grad, FLAGS.max_gradient_norm), var) for grad, var in grads_and_vars]
            self.train_op = train_op.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)


        # record summaries
        with tf.name_scope('Summaries'):
            self.accuracy = tf.reduce_mean(reward)
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("cross_entropy", self.xent)
            tf.summary.scalar("baseline_mse", self.baselines_mse)
            tf.summary.scalar("eligibility", self.eligibility)
            tf.summary.histogram("loglikelihood", tf.reduce_mean(loglik, axis=0)) # zero if not sampling!
            tf.summary.histogram("softmax_predictions", preds)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("advantages", tf.reduce_mean(self.advantages))
            tf.summary.scalar("baseline", tf.reduce_mean(baselines))
            tf.summary.scalar("learning_rate", self.learning_rate)

        if full_summary:
            with tf.name_scope('Summ_Locations'):
                sparse_label = tf.argmax(self.y, axis=1)
                for gl in range(num_glimpses):
                    tf.summary.histogram("loc_means_x" + str(gl+1), loc_means[:, gl, 0])
                    tf.summary.histogram("loc_means_y" + str(gl+1), loc_means[:, gl, 1])

                    # visualize for certain digits
                    if gl != 0: # pass on initial
                        tf.summary.histogram("num0_loc_means_x" + str(gl + 1), tf.boolean_mask(loc_means[:, gl, 0], tf.equal(sparse_label, 0)))
                        tf.summary.histogram("num1_loc_means_x" + str(gl + 1), tf.boolean_mask(loc_means[:, gl, 1], tf.equal(sparse_label, 1)))
                        tf.summary.histogram("num6_loc_means_x" + str(gl + 1), tf.boolean_mask(loc_means[:, gl, 0], tf.equal(sparse_label, 6)))
                        tf.summary.histogram("num9_loc_means_x" + str(gl + 1), tf.boolean_mask(loc_means[:, gl, 1], tf.equal(sparse_label, 9)))

            with tf.name_scope('Summ_Trainable'):
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.name, var)
            with tf.name_scope('Summ_Gradients'):
                for grad, var in grads_and_vars:
                    tf.summary.histogram(var.name + '/gradient', grad)

        self.summary = tf.summary.merge_all()

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        self.saver_best = tf.train.Saver(tf.global_variables(), max_to_keep=1)


        # put glimpses back together in a visualizable format
        with tf.variable_scope('Visualization'):
            self.glimpses_composed = []
            self.downscaled_scales = []
            num_scales = len(FLAGS.scale_sizes)
            scale0 = FLAGS.scale_sizes[0]
            out_sz = FLAGS.scale_sizes[-1]
            channel = FLAGS.img_shape[-1]

            masks, paddings = [], []
            for idx in range(num_scales):
                pad_size = (out_sz - FLAGS.scale_sizes[idx]) // 2
                padding = tf.constant([[0, 0],
                                       [pad_size, out_sz - FLAGS.scale_sizes[idx] - pad_size],
                                       [pad_size, out_sz - FLAGS.scale_sizes[idx] - pad_size],
                                       [0, 0]])

                mask = tf.ones([batch_sz*num_glimpses, FLAGS.scale_sizes[idx], FLAGS.scale_sizes[idx], channel])
                mask = tf.pad(mask, padding, mode='CONSTANT', constant_values=0)

                masks.append(mask)
                paddings.append(padding)


            self.glimpses = tf.reshape(self.glimpses, [batch_sz*num_glimpses, -1])
            glimpse_composed = tf.zeros([batch_sz*num_glimpses, out_sz, out_sz, channel], tf.float32)
            scales = tf.split(self.glimpses, num_scales, axis=1)
            last_mask = tf.zeros([batch_sz*num_glimpses, out_sz, out_sz, channel])

            # to check actual model input. Nesting from out to in: scales, glimpses, batch
            for idx in range(num_scales):
                self.downscaled_scales.append(tf.split(
                    tf.reshape(scales[idx], [batch_sz*num_glimpses, scale0, scale0, channel]),
                    num_glimpses, axis=0))

            # Start with smallest scale, pad up to largest, multiply by (mask - last_mask) indicating area not covered by smaller masks
            for idx in range(num_scales):
                # TODO: DO THIS TRANSFORMATION ONCE OUTSIDE THE LOOP TO GET INDICES, THEN USE tf.gather()
                scales[idx] = tf.reshape(scales[idx], [batch_sz*num_glimpses, scale0, scale0, channel])  # resize_images expects [B,H,W,C] -> add channel for MNIST

                # repeat and tile glimpse to scale size (unfortunately there is no tf.repeat)
                repeats = FLAGS.scale_sizes[idx] // scale0
                scales[idx] = tf.transpose(scales[idx], [0, 3, 1, 2])  # put channels in front

                scales[idx] = tf.reshape(
                    tf.tile(tf.reshape(scales[idx], [batch_sz*num_glimpses, channel, scale0 ** 2, 1]), [1, 1, 1, repeats]),
                    [batch_sz*num_glimpses, channel, scale0, repeats * scale0])
                scales[idx] = tf.reshape(
                    tf.tile(tf.reshape(tf.transpose(scales[idx], [0, 1, 3, 2]),
                                       [batch_sz*num_glimpses, channel, repeats * scale0 ** 2, 1]), [1, 1, 1, repeats]),
                    [batch_sz*num_glimpses, channel, repeats * scale0, repeats * scale0])

                scales[idx] = tf.transpose(scales[idx], [0, 3, 2, 1])  # put channels back

                # alternative, but not identical to what model actually sees:
                # scales[idx] = tf.image.resize_images(scales[idx], 2*[FLAGS.scale_sizes[idx]], method=tf.image.ResizeMethod.BILINEAR)

                glimpse_composed += (masks[idx] - last_mask) * tf.pad(scales[idx], paddings[idx], mode='CONSTANT',
                                                                      constant_values=0.)
                last_mask = masks[idx]

            self.glimpses_composed = tf.split(glimpse_composed, num_glimpses, axis=0)
