import os
import pickle
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import re
import numpy as np
from six.moves import urllib
import tarfile

def cut_data_batch_size(train, valid, test, FLAGS):
    cut_data = []

    for idx, data in enumerate([train, valid, test]):
        x, y = data
        if idx == 0:
            cut = x.shape[0] % FLAGS.batch_size
        else:
            cut = x.shape[0] % (FLAGS.batch_size * FLAGS.MC_samples)
        x = x[:-cut]
        y = y[:-cut]
        cut_data.append((x, y))

    return cut_data[0], cut_data[1], cut_data[2]


def get_cifar(data_dir):
    '''
    Source: http://rohitapte.com/2017/04/22/image-recognition-on-the-cifar-10-dataset-using-deep-learning/
    :param data_dir:
    :return:
    '''
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
    if os.path.isfile(data_file):
        pass
    else:
        def progress(block_num, block_size, total_size):
            progress_info = [cifar10_url, float(block_num * block_size) / float(total_size) * 100.0]
            print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")

        filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
        tarfile.open(filepath, 'r:gz').extractall(data_dir)

    def load_cifar10data(filename):
        with open(filename, mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')
            features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float32)
            labels = batch['labels']
            return features, labels

    x_train = np.zeros(shape=(0, 32, 32, 3), dtype=np.float32)
    train_labels = []
    for i in range(1, 5 + 1):
        ft, lb = load_cifar10data(data_dir + 'cifar-10-batches-py/data_batch_' + str(i))
        x_train = np.vstack((x_train, ft))
        train_labels.extend(lb)

    y_train = tf.one_hot(train_labels, depth=10)

    x_test, test_labels = load_cifar10data(data_dir + 'cifar-10-batches-py/test_batch')
    y_test = tf.one_hot(test_labels, depth=10)

    return (x_train, y_train), (x_test, y_test)


def get_data(FLAGS):
    '''
    :return: train, valid, test, each a tuple of (images, 1hot-labels)
    '''
    train, valid, test = None, None, None

    path = FLAGS.data_dir + FLAGS.dataset + "/"

    if FLAGS.dataset == 'MNIST':
        mnist = input_data.read_data_sets(path, one_hot=True)

        train = (np.reshape(mnist.train.images, [mnist.train.images.shape[0]] + FLAGS.img_shape),  mnist.train.labels)
        valid = (np.reshape(mnist.validation.images, [mnist.validation.images.shape[0]] + FLAGS.img_shape), mnist.validation.labels)
        test = (np.reshape(mnist.test.images, [mnist.test.images.shape[0]] + FLAGS.img_shape), mnist.test.labels)

    elif FLAGS.dataset == "MNIST_cluttered":
        NUM_FILES = 100000
        filenames = np.array([path + "img_{}.png".format(i) for i in range(1, NUM_FILES+1)])
        with open(path + "labels.txt",'r') as file:
            labels = file.read()
        labels = re.sub(r"\d+\t", "", labels).split('\n')
        labels = np.array(labels, dtype=np.int32)
        NUM_CLASSES = labels.max() + 1
        labels = tf.one_hot(labels, depth=NUM_CLASSES)
        labels.set_shape([NUM_FILES, NUM_CLASSES])

        train = (filenames[:80001], labels[:80001])
        valid = (filenames[80001:90001], labels[80001:90001])
        test =  (filenames[90001:], labels[90001:])

    elif FLAGS.dataset == "cifar10":
        (x_train, y_train), test = get_cifar(path)# cifar10.load_data()
        train = (x_train[:-10000], y_train[:-10000])
        valid = (x_train[-10000:], y_train[-10000:])

    elif FLAGS.dataset == "omniglot":
        alphabets = os.listdir(path)
        # completely held out alphabets. Used as test: measure is ration of correctly classified as unknown
        num_test = 1
        # number of alphabets used for unknown label during training
        num_unknown = 1

        alphabets_test = alphabets[:num_test]
        alphabets_train = alphabets[num_test:]
        alphabets_train_unknown = alphabets_train[:num_unknown]

        training = [(os.path.join(path, alpha, character, name), alpha + "_" + character)
                    if alpha not in alphabets_train_unknown
                    else (os.path.join(path, alpha, character, name), "unknown")
                    for alpha in alphabets_train
                    for character in os.listdir(os.path.join(path, alpha))
                    for name in os.listdir(os.path.join(path, alpha, character))]
        # for i, tupl in enumerate(training):
        #     if tupl[1].split("_")[0] in alphabets_train_unknown:
        #         training[i] = (tupl[0], "unknown")

        train = training[:-4000]
        valid = training[-4000:]

        #change list of  tuples into tuple of 2 lists
        train = list(zip(*train))
        valid= list(zip(*valid))


        test = [(os.path.join(path, alpha, character, name), "unknown")
                for alpha in alphabets_test
                for character in os.listdir(os.path.join(path, alpha))
                for name in os.listdir(os.path.join(path, alpha, character))]
        test = list(zip(*test))


    # Cut to a multiple of batch_size. Necessary as the data iterator otherwise will return the rest of the data
    # whenever some observations are left (variable batch size not supported atm).
    # tf.contrib.data.batch_and_drop_remainder would be another option, but this
    # would also require to iterators. When deciding which iterator to use with tf.cond(), both will be executed
    # every time tf.cond() is called.
    train, valid, test = cut_data_batch_size(train, valid, test, FLAGS)

    FLAGS.train_batches_per_epoch = train[0].shape[0] // FLAGS.batch_size
    FLAGS.batches_per_eval_valid  = valid[0].shape[0] // (FLAGS.batch_size * FLAGS.MC_samples)
    FLAGS.batches_per_eval_test   = test[0].shape[0]  // (FLAGS.batch_size * FLAGS.MC_samples)

    return train, valid, test


def parse_function(filename, label, FLAGS):
    if FLAGS.dataset=="MNIST_cluttered":
        image_string = tf.read_file(filename)

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.image.decode_png(image_string, channels=FLAGS.img_shape[-1])

        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        image.set_shape(FLAGS.img_shape)
    else:
        image = filename
    return image, label


def translate_function(image, label, FLAGS):
    '''
    Not sure if translation differs every epoch or not atm.
    Alternative: could pass a vector with pre-sampled x1, y1 (and a counter to index) along to ensure same translation.
    '''
    if FLAGS.translated_size:
        pad_height = FLAGS.translated_size - FLAGS.img_shape[0]
        pad_width = FLAGS.translated_size - FLAGS.img_shape[1]

        image = tf.reshape(image, FLAGS.img_shape)

        y1 = tf.random_uniform(shape=[], maxval=pad_height, dtype=tf.int32)
        x1 = tf.random_uniform(shape=[], maxval=pad_width, dtype=tf.int32)
        image = tf.pad(image, [(y1, pad_height - y1), (x1, pad_width - x1), (0,0)], mode='constant', constant_values=0.)

    return image, label


def pipeline(data, FLAGS, shuffle, train, repeats, preftch=2):
    translate_fn = lambda img, label: translate_function(img, label, FLAGS)
    parse_fn = lambda img, label: parse_function(img, label, FLAGS)

    out_data = (tf.data.Dataset.from_tensor_slices(data)
                .shuffle(buffer_size=tf.cast(data[0].shape[0] / 4, tf.int64), reshuffle_each_iteration=shuffle)
                )
    if FLAGS.dataset in ["MNIST_cluttered", "omniglot"]:
        out_data = (out_data.map(parse_fn, num_parallel_calls=4)
                    )
    if FLAGS.translated_size:
        out_data = out_data.map(translate_fn, num_parallel_calls=4)

    if not train:
        MC = FLAGS.MC_samples
    else:
        MC = 1

    out_data = (out_data
                .batch(FLAGS.batch_size * MC)
                .cache()
                .repeat(repeats)
                .prefetch(preftch)
                )

    return out_data

def input_fn(FLAGS):
    '''train, valid, test: tuples of (images, labels)'''

    # load datasets
    train, valid, test = get_data(FLAGS)

    tr_data    = pipeline(train, FLAGS, train=True, repeats=tf.cast(tf.ceil(FLAGS.num_epochs + FLAGS.num_epochs / FLAGS.eval_step_interval), tf.int64), shuffle=True)
    # repeats * 2 because also used for visualization etc.
    valid_data = pipeline(valid, FLAGS, train=False, repeats=tf.cast(tf.ceil(2 * FLAGS.num_epochs / FLAGS.eval_step_interval), tf.int64), shuffle=False)
    test_data  = pipeline(test, FLAGS, train=False, repeats=2, shuffle=False)

    # adjust flag
    if FLAGS.translated_size:
        FLAGS.img_shape[0:2] = 2 * [FLAGS.translated_size]

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, tr_data.output_types, tr_data.output_shapes)
    # iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
    images, labels = iterator.get_next()

    train_init_op = tr_data.make_one_shot_iterator()
    valid_init_op = valid_data.make_one_shot_iterator()
    test_init_op  = test_data.make_one_shot_iterator()

    # training_init_op = iterator.make_initializer(tr_data)
    # valid_init_op    = iterator.make_initializer(valid_data)
    # test_init_op     = iterator.make_initializer(test_data)

    inputs = {'images': images,
              'labels': labels,
              'handle': handle,
              'train_init_op': train_init_op,
              'valid_init_op': valid_init_op,
              'test_init_op':  test_init_op}
    return inputs