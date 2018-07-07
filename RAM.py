import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from input_fn import input_fn
from Model_modules import RetinaSensor, GlimpseNetwork, GlimpseNetwork_DRAM, LocationNetwork, Rewards, _rnn_cell_RAM
from utility import weight_variable, bias_variable


class RAMNetwork(object):
    def __init__(self, FLAGS, full_summary=False,):
        '''Input:
        img_shape: [H,W,C]
        '''
        tf.reset_default_graph()

        self.size_glimpse_out = FLAGS.size_glimpse_out
        num_glimpses = FLAGS.num_glimpses
        self.num_scales = len(FLAGS.scale_sizes)
        self.patch_shape = [self.num_scales, FLAGS.scale_sizes[0], FLAGS.scale_sizes[0], FLAGS.img_shape[-1]]
        self.patch_shape_flat = np.prod(self.patch_shape)
        self.FLAGS = FLAGS
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        with tf.name_scope('Placeholders'):
            self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        with tf.device('/device:CPU:*'):
            with tf.name_scope('Dataset'):
                inputs = input_fn(FLAGS)
                self.features_ph_train = inputs['features_ph_train']
                self.labels_ph_train   = inputs['labels_ph_train']
                self.features_ph_valid = inputs['features_ph_valid']
                self.labels_ph_valid   = inputs['labels_ph_valid']
                self.features_ph_test  = inputs['features_ph_test']
                self.labels_ph_test    = inputs['labels_ph_test']
                self.handle        = inputs['handle']
                self.train_init_op = inputs['train_init_op']
                self.valid_init_op = inputs['valid_init_op']
                self.test_init_op  = inputs['test_init_op']

                self.x, self.y = (inputs['images'], inputs['labels'])
                (x, y) = (tf.tile(self.x, [FLAGS.MC_samples, 1, 1, 1]),
                          tf.tile(self.y, [FLAGS.MC_samples]))

                batch_sz = tf.shape(x)[0]  # potentially variable batch_size

                img_NHWC = tf.reshape(x, [batch_sz] + FLAGS.img_shape)

        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.maximum(
                                    tf.train.exponential_decay(
                                            FLAGS.learning_rate,
                                            self.global_step,
                                            FLAGS.learning_rate_decay_steps,
                                            FLAGS.learning_rate_decay_factor,
                                            staircase=True),
                                    FLAGS.min_learning_rate)

        location_network = LocationNetwork(img_NHWC, FLAGS)
        retina_sensor    = RetinaSensor(FLAGS)
        if FLAGS.ConvGlimpse:
            glimpse_network = GlimpseNetwork_DRAM(FLAGS, self.patch_shape, self.size_glimpse_out)
        else:
            glimpse_network = GlimpseNetwork(FLAGS, self.patch_shape, self.size_glimpse_out)

        with tf.name_scope('CoreNetwork'):
            if FLAGS.cell == 'RNN':
                cell = _rnn_cell_RAM(FLAGS.size_rnn_state, activation=tf.nn.relu)
            elif FLAGS.cell == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell(FLAGS.size_rnn_state, activation=tf.nn.relu)
            # cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_units=FLAGS.size_rnn_state, num_layers=1)
            locs_ta         = tf.TensorArray(tf.float32, size=num_glimpses, name='locs_ta')
            loc_means_ta    = tf.TensorArray(tf.float32, size=num_glimpses, name='loc_means_ta')
            glimpses_ta = tf.TensorArray(tf.float32, size=num_glimpses, name='glimpses_ta')  # for visualization
            action_ta       = tf.TensorArray(tf.float32, size=num_glimpses, name='action_ta')
            output_ta = (locs_ta, loc_means_ta, glimpses_ta)

            def loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output

                if cell_output is None:  # time == 0
                    loc, loc_mean   = location_network.initial_loc()
                    next_cell_state = cell.zero_state(batch_sz, tf.float32)
                    loop_state      = output_ta
                else:  # time == 1+
                    loc, loc_mean = location_network(cell_output, self.is_training)
                    next_cell_state = cell_state

                img_patch_flat = retina_sensor(img_NHWC, tf.clip_by_value(loc, -1, 1))
                # tf automatically reparametrizes the normal dist., but we don't want to propagate the supervised loss into location
                glimpse = glimpse_network(img_patch_flat, tf.stop_gradient(loc))

                with tf.name_scope('write_or_finished'):
                    elements_finished = (time >= num_glimpses)
                    finished = tf.reduce_all(elements_finished)

                    def _write():
                        return (loop_state[0].write(time, loc),
                                loop_state[1].write(time, loc_mean),
                                loop_state[2].write(time, img_patch_flat))
                    next_loop_state = tf.cond(finished,
                                              lambda: loop_state,
                                              lambda: _write())

                return (elements_finished, glimpse, next_cell_state,
                        emit_output, next_loop_state)

            outputs_ta, final_state, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
            rnn_outputs = outputs_ta.stack(name='stack_rnn_outputs')  # [time, batch_sz, num_cell]

        with tf.name_scope('stack_outputs'):
            self.locs = tf.transpose(loop_state_ta[0].stack(name='stack_locs'), [1,0,2])  # [batch_sz, timesteps, loc_dims]
            loc_means = tf.transpose(loop_state_ta[1].stack(name='stack_loc_means'), [1,0,2])
            self.glimpses = loop_state_ta[2].stack(name='stack_glimpses')

        with tf.variable_scope('Baseline'):
            self.b_W = weight_variable([FLAGS.size_rnn_state, 1], name='b_W')
            self.b_b = bias_variable([1], name='b_b')
            baselines = [tf.squeeze(tf.matmul(tf.stop_gradient(rnn_outputs[i]), self.b_W) + self.b_b) for i in range(num_glimpses - 1)]
            baselines = tf.stack(baselines, axis=1)  # [batch_sz, timesteps]

        # classification after last time-step
        with tf.variable_scope('CoreNetwork_preds'):
            fc_pred = tf.layers.Dense(FLAGS.num_classes, kernel_initializer=xavier_initializer(), name='fc_logits')
            logits = fc_pred(rnn_outputs[-1])
            self.probabilities = tf.nn.softmax(logits)
            self.prediction = tf.argmax(logits, 1)

            # store prediction at each step. Tuple of most likely (class, probability) for each step
            self.intermed_preds = []
            for i in range(num_glimpses):
                p = tf.nn.softmax(fc_pred(tf.stop_gradient(rnn_outputs[i])))
                p_class = tf.argmax(p, 1)
                idx = tf.transpose([tf.cast(tf.range(batch_sz), dtype=tf.int64), p_class])
                self.intermed_preds.append((p_class, tf.gather_nd(p, idx)))

        with tf.name_scope('Cross-entropy_loss'):
            self.xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

        with tf.name_scope('Rewards'):
            self.Rewards = Rewards(FLAGS)
            self.returns, reward, self.unknown_accuracy = self.Rewards(self.prediction, y)

            self.advantages = self.returns - baselines

            with tf.name_scope('loglikelihood'):
                # only want gradients flow through the suggested mean
                # gaussian = tf.distributions.Normal(tmp_mean[:,1:], scale=FLAGS.loc_std)
                # loglik = gaussian._log_prob(tf.stop_gradient(tmp_loc[:,1:]))
                # loglik = tf.reduce_sum(loglik, axis=2)
                z = (tf.stop_gradient(self.locs[:,1:]) - loc_means[:,1:]) / FLAGS.loc_std  # [batch_sz, timesteps, loc_dims]
                loglik = -0.5 * tf.reduce_sum(tf.square(z), axis=2)

            with tf.name_scope('RL_loss'):
                # do not propagate back through advantages
                self.RL_loss = tf.reduce_mean(loglik * tf.stop_gradient(self.advantages))

        with tf.name_scope('Baseline_loss'):
            self.baselines_mse = tf.reduce_mean(tf.square(tf.stop_gradient(self.returns) - baselines))

        with tf.name_scope('Hybrid_loss'):
            self.loss = - FLAGS.learning_rate_RL * self.RL_loss + self.xent + self.baselines_mse

        with tf.variable_scope('Adam'):
            train_op = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = train_op.compute_gradients(self.loss)

            # look at selected gradients
            self.gradient_check = {v: tf.reduce_mean(g) for g, v in grads_and_vars}

            clipped_grads_and_vars = [(tf.clip_by_norm(grad, FLAGS.max_gradient_norm), var) for grad, var in grads_and_vars]
            self.train_op = train_op.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)


        with tf.name_scope('Summaries'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, y), tf.float32))
            probs    = tf.reshape(self.probabilities, [FLAGS.MC_samples, -1, FLAGS.num_classes])
            avg_pred = tf.reduce_mean(probs, axis=0)
            avg_pred = tf.cast(tf.equal(tf.argmax(avg_pred, 1), self.y), tf.float32)
            self.accuracy_MC = tf.reduce_mean(avg_pred, name='accuracy')
            self.reward = tf.reduce_mean(reward, name='avg_reward')
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("cross_entropy", self.xent)
            tf.summary.scalar("baseline_mse", self.baselines_mse)
            tf.summary.scalar("RL_loss", self.RL_loss)
            tf.summary.histogram("loglikelihood", tf.reduce_mean(loglik, axis=0)) # zero if not sampling!
            tf.summary.histogram("softmax_predictions", self.probabilities)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("accuracy_MC", self.accuracy_MC)
            tf.summary.scalar("reward", self.reward)
            tf.summary.scalar("advantages", tf.reduce_mean(self.advantages))
            tf.summary.scalar("baseline", tf.reduce_mean(baselines))
            tf.summary.scalar("learning_rate", self.learning_rate)

        if full_summary:
            with tf.name_scope('Summ_RNN'):
                tf.summary.image('rnn_outputs',
                             tf.reshape(tf.transpose(rnn_outputs, [1, 0, 2]),  # [batch_sz, cells, time]
                                        [-1, FLAGS.size_rnn_state, num_glimpses, 1]),
                             max_outputs=3)
            with tf.name_scope('Summ_Locations'):
                sparse_label = tf.argmax(y, axis=1)
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

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2, name='Saver')
        self.saver_best = tf.train.Saver(tf.global_variables(), max_to_keep=1, name='Saver_best')


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


            self.glimpses_reshpd = tf.reshape(self.glimpses, [batch_sz*num_glimpses, -1])
            glimpse_composed = tf.zeros([batch_sz*num_glimpses, out_sz, out_sz, channel], tf.float32)
            scales = tf.split(self.glimpses_reshpd, num_scales, axis=1)
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

    def setup(self, sess, train_data, valid_data, test_data):
        if self.FLAGS.start_checkpoint:
            self.saver.restore(sess, self.FLAGS.path + "/cp.ckpt")
            start_step = self.global_step.eval(session=sess)
            tf.logging.info('Training from step: %d ', start_step)
        else:
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(self.FLAGS.path + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(self.FLAGS.path + '/valid')
        test_writer  = tf.summary.FileWriter(self.FLAGS.path + '/test')

        train_handle = sess.run(self.train_init_op.string_handle())
        valid_handle = sess.run(self.valid_init_op.string_handle())
        test_handle  = sess.run(self.test_init_op.string_handle())
        sess.run(self.train_init_op.initializer, feed_dict={self.features_ph_train: train_data[0],
                                                             self.labels_ph_train: train_data[1]})
        sess.run(self.valid_init_op.initializer, feed_dict={self.features_ph_valid: valid_data[0],
                                                             self.labels_ph_valid: valid_data[1]})
        sess.run(self.test_init_op.initializer, feed_dict={self.features_ph_test: test_data[0],
                                                            self.labels_ph_test: test_data[1]})

        return train_writer, valid_writer, test_writer, train_handle, valid_handle, test_handle