#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: style_transfer.py
# Project: DeepLearning_FastCampus
# Created Date: 2021-11-25 Thursday 05:07:13
# Author: transpine(transpine@gmail.com)
# ----------------------------------------------
# Last Modified: 2021-11-29 Monday 07:30:20
# Modified By: transpine
# ----------------------------------------------
# Copyright (c) 2021 devfac
# 
# MIT License
# 
# Copyright (c) 2021 devfac
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###

from __future__ import print_function, absolute_import, division, unicode_literals

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

import PIL.Image
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

from devfacPyLogger import log

def tensor_to_image(tensor):
  tensor = tensor * 255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor) > 3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  # https://docs.w3cub.com/tensorflow~python/tf/image/decode_image
  # 여기 채널은 [height, width, num_channles]이므로 폭, 높이만 구해서 scale하여
  # 긴쪽 기준으로 맞춰준다.
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  #img가 tensor이므로 tensorflow의 shape, cast를 사용  
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim
  new_shape = tf.cast(shape*scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]

  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)
  
  plt.imshow(image)

  if title:
    plt.title(title)

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# gram matrix를 정의합니다.
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

  return result/(num_locations)
  
style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30
epochs = 10
steps_per_epoch = 100

content_path = keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg') # https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1,2,1)
imshow(content_image, 'Content Image')
plt.subplot(1,2,2)
imshow(style_image, 'Style Image')
plt.waitforbuttonpress()
plt.close()

vgg = keras.applications.VGG19(include_top=False, weights='imagenet')
for layer in vgg.layers:
  log.info(layer.name)

content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
  vgg = keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = keras.Model([vgg.input], outputs)
  return model

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

for name, output in zip(style_layers, style_outputs):
  log.info(name)
  log.info(f"shape: {output.numpy().shape}")
  log.info(f"min: {output.numpy().min()}")
  log.info(f"max: {output.numpy().max()}")
  log.info(f"mean: {output.numpy().mean()}")

class StyleContentModel(keras.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, input):
    inputs = input * 255.0
    preprocessed_input = keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)

    style_outputs = outputs[:self.num_style_layers]
    content_outputs = outputs[self.num_style_layers:]

    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
    content_dict = {
      content_name : value for content_name, value in zip(self.content_layers, content_outputs)
    }

    style_dict = {
      style_name: value for style_name, value in zip(self.style_layers, style_outputs)
    }

    return {'content': content_dict, 'style': style_dict}

extractor = StyleContentModel(style_layers, content_layers)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

opt = keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

def style_content_loss(outputs):
  style_outputs = outputs['style']
  content_outputs = outputs['content']

  style_loss = tf.add_n([tf.reduce_mean( (style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
  style_loss *= style_weight / num_style_layers


  content_loss = tf.add_n([tf.reduce_mean( (content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
  content_loss *= style_weight / num_style_layers

  loss = style_loss + content_loss

  return loss

def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]

  return x_var, y_var

def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)

  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


@tf.function
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight * total_variation_loss(image)

  grad = tape.gradient(loss, image)
  # 학습해야할 타겟이 네트워크로 여러개의 레이어가 나열되어 있다면 zip()을 사용했겠지만
  # 여기서는 이미지 한장이 타겟이므로 array
  opt.apply_gradients([(grad, image)])
  
  image.assign(clip_0_1(image))

synthetic_image = tf.Variable(content_image)

# epochs 횟수만큼 최적화를 진행합니다.
start = time.time()
step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(synthetic_image)
    print(".", end='')
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))

# 최적화 결과로 생성된 합성 이미지를 화면에 띄우고 파일로 저장합니다.
plt.imshow(tensor_to_image(synthetic_image))
plt.savefig("stylized-image.png")
plt.waitforbuttonpress()