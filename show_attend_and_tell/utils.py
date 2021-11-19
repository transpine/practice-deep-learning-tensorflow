#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: util.py
# Project: show_attend_and_tell
# Created Date: 2021-11-19 Friday 03:33:23
# Author: transpine(transpine@gmail.com)
# ----------------------------------------------
# Last Modified: 2021-11-19 Friday 03:48:10
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

import tensorflow
import tensorflow.keras as keras

def load_image(image_path):
  img = tensorflow.io.read_file(image_path)
  img = tensorflow.image.decode_jpeg(img, channels=3)
  img = tensorflow.image.resize(img, (299, 299))
  # -1과 1 사이의 값으로 정규화(scale)된다.
  img = keras.applications.inception_v3.preprocess_input(img)

  return img, image_path