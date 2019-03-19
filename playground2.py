import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


# path to tfrecord test file
tfrecord_filename = 'C:/Users/erski/Documents/NSynth/nsynth-test.tfrecord'
fs = 16000


def main(filename, batch_size, num_epochs):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    batch = tf.train.batch([serialized_example], batch_size=batch_size)
    parsed_batch = tf.parse_example(batch, features={
        'audio': tf.FixedLenFeature([64000], dtype=tf.float32),
        'pitch': tf.FixedLenFeature([1], dtype=tf.int64),
        'velocity': tf.FixedLenFeature([1], dtype=tf.int64),
        'instrument_str': tf.VarLenFeature(dtype=tf.string)
    })

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners()
        try:
            while True:
                data_batch = sess.run(parsed_batch)
                print(data_batch['instrument_str'].values, data_batch['pitch'], data_batch['audio'][0])
                sd.play(data_batch['audio'][0], fs, blocking=True)
                # data process

        except tf.errors.OutOfRangeError:
            print('out of range')


main(tfrecord_filename, 1, 1)

