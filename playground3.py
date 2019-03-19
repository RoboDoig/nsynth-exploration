import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# path to tfrecord test file
tfrecord_filename = 'C:/Users/erski/Documents/NSynth/nsynth-test.tfrecord'
fs = 16000


# format by which single data records are parsed
def extract_fn(data_record):
    features = {
        'audio': tf.FixedLenFeature([64000], dtype=tf.float32),
        'pitch': tf.FixedLenFeature([1], dtype=tf.int64),
        'velocity': tf.FixedLenFeature([1], dtype=tf.int64),
        'instrument_str': tf.VarLenFeature(dtype=tf.string)
    }
    sample = tf.parse_single_example(data_record, features)
    return sample


# prepare the dataset, read from file, map on the parsing function, shuffle and batch, make iterator
dataset = tf.data.TFRecordDataset([tfrecord_filename])
dataset = dataset.map(extract_fn)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


# run session, print out first record in each batch an play corresponding sample
with tf.Session() as sess:
    try:
        while True:
            batch = sess.run(next_element)
            print(batch['instrument_str'].values[0], batch['pitch'][0], batch['audio'][0])
            sd.play(batch['audio'][0], fs, blocking=True)
    except tf.errors.OutOfRangeError:
        pass
