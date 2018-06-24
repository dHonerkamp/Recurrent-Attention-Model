import os
import pickle
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import re
import numpy as np
from six.moves import urllib
import tarfile


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
            features /= 255
            labels = batch['labels']
            return features, labels

    x_train = np.zeros(shape=(0, 32, 32, 3), dtype=np.float32)
    train_labels = []
    for i in range(1, 5 + 1):
        ft, lb = load_cifar10data(data_dir + 'cifar-10-batches-py/data_batch_' + str(i))
        x_train = np.vstack((x_train, ft))
        train_labels.extend(lb)

    # y_train = tf.one_hot(train_labels, depth=10)
    y_train = np.array(train_labels, dtype=np.int64)

    x_test, test_labels = load_cifar10data(data_dir + 'cifar-10-batches-py/test_batch')
    # y_test = tf.one_hot(test_labels, depth=10)
    y_test = np.array(test_labels, dtype=np.int64)

    return (x_train, y_train), (x_test, y_test)


def get_data(FLAGS):
    '''
    :return: train, valid, test, each a tuple of (images, sparse labels)
    '''
    train, valid, test = None, None, None

    data_path = FLAGS.data_dir + FLAGS.dataset + "/"

    if FLAGS.dataset == 'MNIST':
        mnist = input_data.read_data_sets(data_path, one_hot=False)
        FLAGS.num_classes = 10

        train = (np.reshape(mnist.train.images, [mnist.train.images.shape[0]] + FLAGS.img_shape),  np.array(mnist.train.labels, dtype=np.int64))
        valid = (np.reshape(mnist.validation.images, [mnist.validation.images.shape[0]] + FLAGS.img_shape), np.array(mnist.validation.labels, dtype=np.int64))
        test  = (np.reshape(mnist.test.images, [mnist.test.images.shape[0]] + FLAGS.img_shape), np.array(mnist.test.labels, dtype=np.int64))

    elif FLAGS.dataset == "MNIST_cluttered":
        NUM_FILES = 100000
        filenames = np.array([data_path + "img_{}.png".format(i) for i in range(1, NUM_FILES+1)])
        with open(data_path + "labels.txt",'r') as file:
            labels = file.read()
        labels = re.sub(r"\d+\t", "", labels).split('\n')
        labels = np.array(labels, dtype=np.int64)
        FLAGS.num_classes = labels.max() + 1

        train = (filenames[:80001], labels[:80001])
        valid = (filenames[80001:90001], labels[80001:90001])
        test =  (filenames[90001:], labels[90001:])

    elif FLAGS.dataset == "cifar10":
        (x_train, y_train), test = get_cifar(data_path)# cifar10.load_data()
        train = (x_train[:-10000], y_train[:-10000])
        valid = (x_train[-10000:], y_train[-10000:])

        FLAGS.num_classes = 10

    elif FLAGS.dataset == "omniglot":
        data_path = os.path.join(data_path, "images_all")
        alphabets = sorted(os.listdir(data_path))

        alphabets_test  = alphabets[:FLAGS.n_unknown_test]
        alphabets_train = alphabets[FLAGS.n_unknown_test:]

        if FLAGS.n_unknown_train:
            alphabets_train_unknown = alphabets[FLAGS.n_unknown_test:FLAGS.n_unknown_test+FLAGS.n_unknown_train]
        else:
            alphabets_train_unknown = []

        labels = [alpha + "_" + character
                  if (alpha in alphabets_train) and (alpha not in alphabets_train_unknown)
                  else "unknown"
                  for alpha in alphabets
                  for character in os.listdir(os.path.join(data_path, alpha))]

        labels_dict = {}
        for i, a in enumerate(sorted(set(labels))):
            labels_dict[a] = i
        FLAGS.num_classes = np.max(list(labels_dict.values())) + 1  # first class is 0
        FLAGS.unknown_label = labels_dict["unknown"]

        training = np.array([(os.path.join(data_path, alpha, character, name), labels_dict[alpha + "_" + character])
                    if alpha not in alphabets_train_unknown
                    else (os.path.join(data_path, alpha, character, name), labels_dict["unknown"])
                    for alpha in alphabets_train
                    for character in os.listdir(os.path.join(data_path, alpha))
                    for name in os.listdir(os.path.join(data_path, alpha, character))])

        n = len(training)
        val_len = min(int(n * 0.2), 4000)
        np.random.seed(42)
        # store permutation as seed does not ensure cross-platform compatibility
        if FLAGS.start_checkpoint:
            idx = np.load(FLAGS.path + '/random_idx.npy')
        else:
            idx = np.random.permutation(n)
            os.makedirs(FLAGS.path)  # o/w create from summary writer
            np.save(FLAGS.path + '/random_idx.npy', idx)

        train = training[idx[:-val_len]]
        valid = training[idx[-val_len:]]

        #change list of  tuples into tuple of 2 lists
        def to_data_tuple(l):
            (a, b) = map(list, zip(*l))
            return (np.array(a), np.array(b, dtype=np.int64))
        train = to_data_tuple(train)
        valid = to_data_tuple(valid)

        test = [(os.path.join(data_path, alpha, character, name), labels_dict["unknown"])
                for alpha in alphabets_test
                for character in os.listdir(os.path.join(data_path, alpha))
                for name in os.listdir(os.path.join(data_path, alpha, character))]
        test = to_data_tuple(test)

        print("Num labels: {}, unknown label: {}".format(len(set(labels)), labels_dict["unknown"]))
        print("Num alphabets train: {}, unknown train: {}, test: {}".
              format(len(alphabets_train), FLAGS.n_unknown_train, len(alphabets_test)))

    print("Obs per dataset: ", len(train[0]), len(valid[0]), len(test[0]))


    FLAGS.train_batches_per_epoch = train[0].shape[0] // FLAGS.batch_size
    FLAGS.batches_per_eval_valid  = valid[0].shape[0] // FLAGS.batch_size
    FLAGS.batches_per_eval_test   = test[0].shape[0]  // FLAGS.batch_size

    return train, valid, test


def parse_function(filename, label, FLAGS):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_png(image_string, channels=FLAGS.img_shape[-1])

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape(FLAGS.img_shape)

    if FLAGS.dataset == "omniglot":
        # change lines to 1 and empty space to 0 as in mnist to be potentially able to cross-train
        image = tf.abs(image - 1)
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


def pipeline(data, FLAGS, shuffle, repeats, preftch=2):
    translate_fn = lambda img, label: translate_function(img, label, FLAGS)
    parse_fn = lambda img, label: parse_function(img, label, FLAGS)

    out_data = (tf.data.Dataset.from_tensor_slices(data)
                .shuffle(buffer_size=tf.cast(data[0].shape[0], tf.int64), reshuffle_each_iteration=shuffle)
                )
    if FLAGS.dataset in ["MNIST_cluttered", "omniglot"]:
        out_data = out_data.map(parse_fn, num_parallel_calls=4)
    if FLAGS.translated_size:
        out_data = out_data.map(translate_fn, num_parallel_calls=4)

    out_data = (out_data
                .batch(FLAGS.batch_size)
                # .cache()
                .repeat(repeats)
                .prefetch(preftch)
                )

    return out_data

def input_fn(FLAGS):
    '''train, valid, test: tuples of (images, labels)'''

    features_ph_train = tf.placeholder(FLAGS.data_dtype[0], FLAGS.train_data_shape[0])
    labels_ph_train   = tf.placeholder(FLAGS.data_dtype[1], FLAGS.train_data_shape[1])
    features_ph_valid = tf.placeholder(FLAGS.data_dtype[0], FLAGS.valid_data_shape[0])
    labels_ph_valid   = tf.placeholder(FLAGS.data_dtype[1], FLAGS.valid_data_shape[1])
    features_ph_test  = tf.placeholder(FLAGS.data_dtype[0], FLAGS.test_data_shape[0])
    labels_ph_test    = tf.placeholder(FLAGS.data_dtype[1], FLAGS.test_data_shape[1])

    tr_data    = pipeline((features_ph_train, labels_ph_train), FLAGS, repeats=tf.cast(tf.ceil(FLAGS.num_epochs + FLAGS.num_epochs / FLAGS.eval_step_interval), tf.int64), shuffle=True)
    # repeats * 2 because also used for visualization etc.
    valid_data = pipeline((features_ph_valid, labels_ph_valid), FLAGS, repeats=tf.cast(tf.ceil(2 * FLAGS.num_epochs / FLAGS.eval_step_interval), tf.int64), shuffle=False)
    test_data  = pipeline((features_ph_test, labels_ph_test), FLAGS, repeats=3, shuffle=False)
    if FLAGS.translated_size:
        FLAGS.img_shape[0:2] = 2 * [FLAGS.translated_size]

    handle = tf.placeholder(tf.string, shape=[], name='handle')
    iterator = tf.data.Iterator.from_string_handle(handle, tr_data.output_types, tr_data.output_shapes)
    # iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
    images, labels = iterator.get_next()

    train_init_op = tr_data.make_initializable_iterator()
    valid_init_op = valid_data.make_initializable_iterator()
    test_init_op  = test_data.make_initializable_iterator()

    # train_init_op = iterator.make_initializer(tr_data)
    # valid_init_op = iterator.make_initializer(valid_data)
    # test_init_op  = iterator.make_initializer(test_data)

    inputs = {'images': images,
              'labels': labels,
              'features_ph_train': features_ph_train,
              'labels_ph_train'  : labels_ph_train,
              'features_ph_valid': features_ph_valid,
              'labels_ph_valid'  : labels_ph_valid,
              'features_ph_test' : features_ph_test,
              'labels_ph_test'   : labels_ph_test,
              'handle': handle,
              'train_init_op': train_init_op,
              'valid_init_op': valid_init_op,
              'test_init_op':  test_init_op}
    return inputs