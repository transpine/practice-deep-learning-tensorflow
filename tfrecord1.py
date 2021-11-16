#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display as display


# In[ ]:


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# In[ ]:


sys.path


# In[ ]:


# sys.path.append('..')


# In[ ]:


IMAGE_PATH = 'data/'


# In[ ]:


def get_image_binary(filename):
  image = Image.open(filename)
  image_arr = np.asarray(image, np.uint8)
  shape = np.array(image_arr.shape, np.int32)
  print(image_arr.shape)
  return shape.tobytes(), image.tobytes()


# In[ ]:


get_image_binary('data/sample.jpg')[0]


# In[ ]:


def main():
  label = 1
  image_file = IMAGE_PATH + 'sample.jpg'
  tfrecord_file = IMAGE_PATH + 'sample.tfrecord'
  # write_tfrecord(label, image_file, tfrecord_file)
  read_tfrecord(tfrecord_file)


# In[ ]:


def write_tfrecord(label, image_file, tfrecord_file):
  shape, binary_image = get_image_binary(image_file)
  write_to_tfrecord(label, shape, binary_image, tfrecord_file)


# In[ ]:


def write_to_tfrecord(label, shape, binary_image, tfrecord_file):
  # writer = tf.python_io.TFRecordWriter(tfrecord_file) #TF1
  # writer = tf.io.TFRecordWriter(tfrecord_file)  #TF2
  # example = tf.train.Example(features=tf.train.Features(feature={
  #   'label': _int64_feature(label),
  #   'shape': _bytes_feature(shape),
  #   'image': _bytes_feature(binary_image)
  # }))
  # writer.write(example.SerializeToString())
  # writer.close()
  with tf.io.TFRecordWriter(tfrecord_file) as writer:
    example = tf.train.Example(features=tf.train.Features(feature={
      'label': _int64_feature(label),
      'shape': _bytes_feature(shape),
      'image': _bytes_feature(binary_image)
    }))
    writer.write(example.SerializeToString())


# In[ ]:


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# In[ ]:


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# In[ ]:


def read_tfrecord(tfrecord_file):
  # label, shape, image = read_from_tfrecord([tfrecord_file])
  read_from_tfrecord([tfrecord_file])

  # with tf.Session() as sess:
  #   coord = tf.train.Coordinator()
  #   threads = tf.train.start_queue_runner(coord=coord)
  #   label, image, shape = sess.run([label, image, shape])
  #   coord.request_stop()
  #   coord.join(threads)
  
  # print(label)
  # print(shape)
  # plt.imshow(image)
  # plt.show()


# In[ ]:


def _parse_image_function(example_proto):
  
  image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'shape': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string)
  }

  return tf.io.parse_single_example(example_proto, image_feature_description)


# In[ ]:


def read_from_tfrecord(tfrecord_file):
  raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
  parsed_image_dataset = raw_dataset.map(_parse_image_function)

  for image_features in parsed_image_dataset:
    # image_raw = image_features['image'].numpy()
    image_raw = tf.io.decode_raw(image_features['image'], tf.uint8)
    image_shape = tf.io.decode_raw(image_features['shape'], tf.int32)
  print('shape: %s' % image_shape)
  image = tf.reshape(image_raw, image_shape)
  # display.display(display.Image(data=image))
  plt.imshow(image)
  plt.show()

  # for raw_record in raw_dataset.take(1):
  #   example = tf.train.Example()
  #   example.ParseFromString(raw_record.numpy())

  #   print(example)

  


# In[ ]:


main()


# In[ ]:




