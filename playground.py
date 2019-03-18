import tensorflow as tf
import numpy as np

# path to tfrecord test file
tfrecord_filename = 'C:/Users/erski/Documents/NSynth/nsynth-test.tfrecord'

sess = tf.InteractiveSession()

# read file
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer([tfrecord_filename])

_, serialized_example = reader.read(filename_queue)

# define features
read_features = {
    'audio': tf.VarLenFeature(dtype=tf.float32),
    'pitch': tf.FixedLenFeature([1], dtype=tf.int64)
}

# extract
read_data = tf.parse_single_example(serialized=serialized_example, features=read_features)

tf.train.start_queue_runners(sess)

# print
for name, tensor in read_data.items():
    print('{}: {}'.format(name, tensor.eval()))
    # print(tensor.values.eval())


