#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: tensor_board.py
# Project: DeepLearning_FastCampus
# Created Date: 2021-11-29 Monday 06:18:25
# Author: transpine(transpine@gmail.com)
# ----------------------------------------------
# Last Modified: 2021-11-29 Monday 07:33:20
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

import tensorflow as tf
import tensorflow.keras as keras

class LinearRegression(keras.Model):
  def __init__(self):
    super(LinearRegression, self).__init__()
    self.linear_layer = keras.layers.Dense(1, activation=None)

  def call(self, x):
    y_pred = self.linear_layer(x)

    return y_pred

@tf.function
def mse_loss(y_pred, y):
  return tf.reduce_mean( tf.square(y_pred - y))

optimizer = tf.optimizers.SGD(0.01)

summary_writer = tf.summary.create_file_writer('./tensorboard_logs')

@tf.function
def train_step(model, x, y):
  with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = mse_loss(y_pred, y)
  
  with summary_writer.as_default():
    tf.summary.scalar('loss', loss, step=optimizer.iterations)
  gradient = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradient, model.trainable_variables))


x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [2.0, 4.0, 6.0, 8.0]

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().batch(1)
train_data_iter = iter(train_data)

LinearRegression_model = LinearRegression()

for i in range(1000):
  batch_xs, batch_ys = next(train_data_iter)
  batch_xs = tf.expand_dims(batch_xs, 0)
  train_step(LinearRegression_model, batch_xs, batch_ys)

x_test = [3.5, 5.0, 5.5, 6.0]
test_data = tf.data.Dataset.from_tensor_slices((x_test))
test_data = test_data.batch(1)

for batch_x_test in test_data:
  batch_x_test = tf.expand_dims(batch_x_test, 0)

  print(tf.squeeze(LinearRegression_model(batch_x_test), 0).numpy())



