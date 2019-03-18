import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np

# path to tfrecord test file
tfrecord_filename = 'C:/Users/erski/Documents/NSynth/nsynth-test.tfrecord'


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'audio': tf.VarLenFeature(dtype=tf.float32),
            'pitch': tf.FixedLenFeature([1], dtype=tf.int64)
        }
    )

    audio = tf.cast(features['audio'], tf.float32)
    pitch = tf.cast(features['pitch'], tf.int64)
    return audio, pitch


def inputs(filename, batch_size, num_epochs):
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)

        dataset.map(decode)

        dataset = dataset.shuffle(1000 + 3 * batch_size)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def main(filename, batch_size, num_epochs):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    batch = tf.train.batch([serialized_example], batch_size=batch_size)
    parsed_batch = tf.parse_example(batch, features={
        'audio': tf.VarLenFeature(dtype=tf.float32),
        'pitch': tf.FixedLenFeature([1], dtype=tf.int64),
        'velocity': tf.FixedLenFeature([1], dtype=tf.int64)
    })

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners()
        try:
            while True:
                data_batch = sess.run(parsed_batch)
                # print(data_batch['audio'].values)
                print(data_batch)
                # data process
                break
        except tf.errors.OutOfRangeError:
            print('out of range')


main(tfrecord_filename, 100, 1)

