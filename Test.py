import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import RAMNetwork
from utility import Utility
from main import store_visualization
from input_fn import input_fn
from tensorflow.examples.tutorials.mnist import input_data
import time
from tensorflow.python.client import timeline

def test_tranlsatedMNIST():
    FLAGS, _ = Utility.parse_arg()

    FLAGS.dataset = 'MNIST'
    train, valid, test = get_data(FLAGS)
    print(train[0].shape)

    FLAGS.dataset = 'translated_MNIST'
    FLAGS.translated_size = 60
    train_trans, valid_trans, test_trans = get_data(FLAGS)
    print(train_trans[0].shape)

    plt.subplot(221)
    plt.imshow(train[0][0].reshape([28, 28]))
    plt.subplot(222)
    plt.imshow(train_trans[0][0].reshape([FLAGS.translated_size, FLAGS.translated_size]))
    plt.subplot(223)
    plt.imshow(train[0][1].reshape([28, 28]))
    plt.subplot(224)
    plt.imshow(train_trans[0][1].reshape([FLAGS.translated_size, FLAGS.translated_size]))
    plt.show()


def train_test_separation():
    FLAGS, _ = Utility.parse_arg()
    FLAGS.batch_size = 5
    FLAGS.MC_samples = 1

    train = (np.zeros(5), ['train'] * 5)
    valid = (np.zeros(5), ['valid'] * 5)
    test  = (np.zeros(5), ['test'] * 5)

    inputs = input_fn(train, valid, test, FLAGS)
    training_init_op = inputs['training_init_op']
    validation_init_op = inputs['valid_init_op']
    test_init_op = inputs['test_init_op']

    x = inputs['images']
    y = inputs['labels']

    with tf.Session() as sess:
        sess.run(training_init_op)
        x_train, y_train = sess.run([x,y])
        print('Train:', y_train)

        sess.run(validation_init_op)
        x_valid, y_valid = sess.run([x, y])
        print('Validation:', y_valid)

        sess.run(test_init_op)
        x_test, y_test = sess.run([x, y])
        print('Test:', y_test)


def test_stack_glimpses_visualization():
    FLAGS, _ = Utility.parse_arg()
    FLAGS.batch_size = 10
    FLAGS.MC_samples = 1

    FLAGS.dataset = 'translated_MNIST'
    FLAGS.translated_size = 60
    FLAGS.scale_sizes = [8, 16, 32]

    train, valid, test = get_data(FLAGS)

    RAM = RAMNetwork(FLAGS=FLAGS,
                     train=train,
                     valid=valid,
                     test=test,
                     patch_shape=len(FLAGS.scale_sizes) * np.power(FLAGS.scale_sizes[0], 2),
                     num_glimpses=FLAGS.num_glimpses,
                     batch_size=FLAGS.batch_size * FLAGS.MC_samples,
                     full_summary=False)

    with tf.Session() as sess:
        sess.run(RAM.test_init_op)

        sess.run(tf.global_variables_initializer())
        x, scales, stacked_glimpses = sess.run([RAM.x, RAM.downscaled_scales, RAM.glimpses_composed], feed_dict={RAM.is_training: False})

        pic = 4
        gl = 2
        shp = [32,32]
        shp_input = 2*[FLAGS.translated_size]

        plt.subplot(221)
        plt.imshow(x[pic].reshape(shp_input))
        plt.title('Scales')
        plt.subplot(222)
        plt.imshow(scales[0][gl][pic].squeeze())
        plt.subplot(223)
        plt.imshow(scales[1][gl][pic].squeeze())
        plt.subplot(224)
        plt.imshow(scales[2][gl][pic].squeeze())
        plt.tight_layout()
        plt.show()

        plt.subplot(221)
        plt.imshow(x[pic].reshape(shp_input))
        plt.title('Composed glimpse')
        plt.subplot(222)
        plt.imshow(stacked_glimpses[gl][pic].reshape(shp))
        plt.subplot(223)
        plt.imshow(x[pic+1].reshape(shp_input))
        plt.subplot(224)
        plt.imshow(stacked_glimpses[gl][pic+1].reshape(shp))
        plt.tight_layout()
        plt.show()

        plt.subplot(221)
        plt.imshow(x[pic+2].reshape(shp_input))
        plt.title('Composed glimpse')
        plt.subplot(222)
        plt.imshow(stacked_glimpses[gl][pic+2].reshape(shp))
        plt.subplot(223)
        plt.imshow(x[pic+3].reshape(shp_input))
        plt.subplot(224)
        plt.imshow(stacked_glimpses[gl][pic+3].reshape(shp))
        plt.tight_layout()
        plt.show()


def test_take_only1Glimpse():
    '''
    For batch_sz 128, 1 epoch train steps, without dataset creation:
        Resizing 1 glimpse: 199.98
        Loop over extract_glimpse: 197.44
    '''
    FLAGS, _ = Utility.parse_arg()
    FLAGS.batch_size = 128
    FLAGS.MC_samples = 10

    FLAGS.dataset = 'MNIST'
    FLAGS.img_shape = [28,28,1]
    FLAGS.translated_size = 60
    FLAGS.scale_sizes = [8, 16, 32]

    RAM = RAMNetwork(FLAGS=FLAGS,
                     patch_shape=len(FLAGS.scale_sizes) * np.power(FLAGS.scale_sizes[0], 2),
                     num_glimpses=FLAGS.num_glimpses,
                     batch_size=FLAGS.batch_size * FLAGS.MC_samples,
                     full_summary=False)

    with tf.Session() as sess:
        train_handle = sess.run(RAM.train_init_op.string_handle())
        t = time.time()

        sess.run(tf.global_variables_initializer())
        epochs = 3
        for i in range(epochs):
            for _ in range(FLAGS.train_batches_per_epoch):
                sess.run([RAM.train_op], feed_dict={RAM.is_training: True,
                                                    RAM.handle: train_handle})
            print(i)
        print("avg. t per epoch: {:.2f}s".format((time.time() - t) / epochs))


def translate_data():
    FLAGS, _ = Utility.parse_arg()
    FLAGS.batch_size = 128
    FLAGS.MC_samples = 10

    # FLAGS.dataset = 'MNIST'
    # FLAGS.translated_size = 100
    FLAGS.scale_sizes = [8, 16, 32]

    # train, valid, test = get_data(FLAGS)
    # train, valid, test = cut_data_batch_size(train, valid, test, FLAGS)
    #
    # FLAGS.train_batches_per_epoch = train[0].shape[0] // FLAGS.batch_size

    RAM = RAMNetwork(FLAGS=FLAGS,
                     patch_shape=len(FLAGS.scale_sizes) * np.power(FLAGS.scale_sizes[0], 2) * FLAGS.img_shape[-1],
                     num_glimpses=FLAGS.num_glimpses,
                     batch_size=FLAGS.batch_size * FLAGS.MC_samples,
                     full_summary=False)

    with tf.Session() as sess:
        sess.run(RAM.training_init_op)

        print("time elapsed: {:.2f}s".format(time.time() - start_time))

        sess.run(tf.global_variables_initializer())
        # for _ in range(FLAGS.train_batches_per_epoch):
        x = sess.run(RAM.x, feed_dict={RAM.is_training: True})

        print("time elapsed: {:.2f}s".format(time.time() - start_time))

        plt.imshow(x[0].reshape(np.squeeze(FLAGS.img_shape)))
        plt.show()


def track_time():
    FLAGS, _ = Utility.parse_arg()
    FLAGS.batch_size = 128
    FLAGS.MC_samples = 10

    FLAGS.dataset = 'MNIST_cluttered'
    FLAGS.img_shape = [100,100,1]
    FLAGS.translated_size = 0
    FLAGS.scale_sizes = [12,24,48]

    if FLAGS.translated_size:
        t_sz = str(FLAGS.translated_size)
    else:
        t_sz = ""
    FLAGS.path = FLAGS.summaries_dir + '/' + FLAGS.dataset + t_sz + '/' + "Test"

    RAM = RAMNetwork(FLAGS=FLAGS,
                     patch_shape=len(FLAGS.scale_sizes) * np.power(FLAGS.scale_sizes[0], 2),
                     num_glimpses=FLAGS.num_glimpses,
                     batch_size=FLAGS.batch_size * FLAGS.MC_samples,
                     full_summary=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_handle = sess.run(RAM.train_init_op.string_handle())
        train_writer = tf.summary.FileWriter(FLAGS.path + '/train', sess.graph)

        for i in range(25):
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            summary, _ = sess.run([RAM.summary, RAM.train_op],
                     options=run_options,
                     run_metadata=run_metadata,
                     feed_dict={RAM.is_training: True,
                                RAM.handle: train_handle})

            train_writer.add_run_metadata(run_metadata, 'step{}'.format(i))
            train_writer.add_summary(summary, i)

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(FLAGS.path + '/train/timeline.json', 'w') as f:
            f.write(ctf)


if __name__ == "__main__":
    start_time = time.time()

    # train_test_separation()
    #
    # test_tranlsatedMNIST()

    # test_stack_glimpses_visualization()

    test_take_only1Glimpse()

    # track_time()

    # translate_data()

    print("time elapsed: {:.2f}s".format(time.time() - start_time))