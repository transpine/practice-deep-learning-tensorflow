#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import IPython.display as display
import cv2
import time

# In[ ]:

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# In[ ]:

IMAGE_PATH = 'data/'

# In[ ]:


def get_image_binary(filename):
  # image = Image.open(filename)
  # image_arr = np.asarray(image, np.uint8)
  # shape = np.array(image_arr.shape, np.int32)
  # print(image_arr.shape)
  # return shape.tobytes(), image.tobytes()
  image_string = open(filename, 'rb').read()
  shape = tf.io.decode_jpeg(image_string).shape
  return np.array(shape, np.int32).tobytes(), image_string

# In[ ]:


# get_image_binary('data/sample.jpg')[0]



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
    # 공식홈에서 소개된 방식. 거의 이미지 사이즈만 차지함.
    image_raw = image_features['image'].numpy()

    ## stanford예제. 사이즈가 매우 크다.
    # image_raw = tf.io.decode_raw(image_features['image'], tf.uint8)

    image_shape = tf.io.decode_raw(image_features['shape'], tf.int32)

  print('shape: %s' % image_shape)

  ## stanford예제. byte를 decode하여 reshape하는 방식
  # image = tf.reshape(image_raw, image_shape)
  # plt.imshow(image)
  # plt.show()

  ## 공식홈에서 소개된 방식. notebook에서 표시
  # display.display(display.Image(data=image_raw))

  ## 이미지 직접 열어 표시 - PIL
  # image = Image.open(io.BytesIO(image_raw))
  # image.show()

  ## 이미지 직접 열어 표시 - opencv 방식 훨씬 빠르다.
  ## https://ballentain.tistory.com/50
  encoded_img = np.frombuffer(image_raw, dtype = np.uint8)
  cv_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
  cv2.imshow('testimage', cv_img)

  ## 스트링값 출력 해보기
  # for raw_record in raw_dataset.take(1):
  #   example = tf.train.Example()
  #   example.ParseFromString(raw_record.numpy())

  #   print(example)

# In[ ]:

def main():
  time_check = time.time()

  label = 1
  image_file = IMAGE_PATH + 'sample.jpg'
  tfrecord_file = IMAGE_PATH + 'sample.tfrecord'
  # write_tfrecord(label, image_file, tfrecord_file)
  read_tfrecord(tfrecord_file)

  print('Time check:', time.time() - time_check)
  
  cv2.waitKey()


# In[ ]:


main()


# In[ ]:




