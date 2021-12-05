# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

# tf.keras.Model을 이용해서 Attention 모델을 정의합니다.
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # sum 이후에 context_vector shape == (batch_size, embedding_dim)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

# tf.keras.Model을 이용해서 CNN Encoder 모델을 정의합니다.
class CNN_Encoder(tf.keras.Model):
  # 이미 Inception v3 모델로 특징 추출된 Feature map이 인풋으로 들어오기 때문에
  # Fully connected layer를 이용한 Embedding만 수행합니다.
  def __init__(self, embedding_dim):
    super(CNN_Encoder, self).__init__()
    # shape after fc == (batch_size, 64, embedding_dim)
    self.fc = tf.keras.layers.Dense(embedding_dim)

  def call(self, x):
    x = self.fc(x)
    x = tf.nn.relu(x)
    return x

# tf.keras.Model을 이용해서 RNN Decoder 모델을 정의합니다.
class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # attention은 별도의 모델로 정의합니다.
    context_vector, attention_weights = self.attention(features, hidden)

    # embedding 이후에 x shape == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # concatenation 이후에 x shape == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # concatenated vector를 GRU에 넣습니다.
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


# #!/usr/bin/env python3
# # -*- coding:utf-8 -*-
# ###
# # File: model.py
# # Project: show_attend_and_tell
# # Created Date: 2021-11-21 Sunday 05:47:27
# # Author: transpine(transpine@gmail.com)
# # ----------------------------------------------
# # Last Modified: 2021-11-23 Tuesday 12:33:03
# # Modified By: transpine
# # ----------------------------------------------
# # Copyright (c) 2021 devfac
# # 
# # MIT License
# # 
# # Copyright (c) 2021 devfac
# # 
# # Permission is hereby granted, free of charge, to any person obtaining a copy of
# # this software and associated documentation files (the "Software"), to deal in
# # the Software without restriction, including without limitation the rights to
# # use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# # of the Software, and to permit persons to whom the Software is furnished to do
# # so, subject to the following conditions:
# # 
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# # 
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
# ###

# import tensorflow as tf
# import tensorflow.keras as keras

# class BahdanauAttention(keras.Model):
#   def __init__(self, units):
#     super(BahdanauAttention, self).__init__()
#     self.W1 = keras.layers.Dense(units)
#     self.W2 = keras.layers.Dense(units)
#     self.V = keras.layers.Dense(1)

#   def call(self, features, hidden):
#     hidden_with_time_axis = tf.expand_dims(hidden, 1)

#     score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
#     attention_weights = tf.nn.softmax(self.V(score), axis=1)

#     context_vector = attention_weights * features
#     context_vector = tf.reduce_sum(context_vector, axis=1)

#     return context_vector, attention_weights

# class CNN_Encoder(keras.Model):
#   def __init__(self, embedding_dim):
#     super(CNN_Encoder, self).__init__()
#     self.fc = keras.layers.Dense(embedding_dim)

#   def call(self, x):
#     x = self.fc(x)
#     x = tf.nn.relu(x)

#     return x

# class RNN_Decoder(keras.Model):
#   def __init__(self, embedding_dim, units, vocab_size):
#     super(RNN_Decoder, self).__init__()

#     self.units = units

#     self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
#     self.gru = keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
#     self.fc1 = keras.layers.Dense(self.units)
#     self.fc2 = keras.layers.Dense(vocab_size)

#     self.attention = BahdanauAttention(self.units)

#   def call(self, x, features, hidden):
#     context_vector, attention_weights = self.attention(features, hidden)

#     x = self.embedding(x)

#     x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    
#     output, state = self.gru(x)

#     x = self.fc1(output)
#     x = tf.reshape(x, (-1, x.reshape[2]))

#     x = self.fc2(x)

#     return x, state, attention_weights

#   def reset_state(self, batch_size):
#     return tf.zeros((batch_size, self.units))
