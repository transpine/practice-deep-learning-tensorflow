# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.backend import sparse_categorical_crossentropy
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 4
BUFFER_SIZE = 10000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
OUTPUT_CHANNELS = 3
learning_rate = 0.001
EPOCHS = 20

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128) )
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128) )

  if tf.random.uniform(()) > .5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1

  return input_image, input_mask

# Input Image, Ground-Truth, Prediction을 화면에 띄웁니다.
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def upsample(filters, size, norm_type="batchnorm", apply_dropout=False):
  """인풋에 대한 Upsample을 수행합니다.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: 필터 개수
    size: 필터 크기
    norm_type: Normalization 종류; 'batchnorm' 혹은 'instancenorm'.
    apply_dropout: True라면 dropout 레이어를 추가합니다.
  Returns:
    Upsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)
  result = keras.Sequential()
  result.add(keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

  #https://stackoverflow.com/questions/68088889/how-to-add-instancenormalization-on-tensorflow-keras
  if norm_type.lower() == "batchnorm":
    result.add(keras.layers.BatchNormalization(axis=1))
  elif norm_type.lower() == "instancenorm":
    # result.add(InstanceNormalization())
    result.add(keras.layers.BatchNormalization(axis=[0, 1]))

  if apply_dropout:
    result.add(keras.layers.ReLU())

  result.add(keras.layers.ReLU())

  return result

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]

  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])

  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])    

             
#dataset api는 학습을 하는 동시에 GPU메모리로 데이터를 옮기는걸 같이하게 해줌.
#num_parallel_calls : 데이터 수행시간을 줄이기 위해 produce, consumer 처럼 데이터 처리
#dataset api는 데이터 준비와 사용 최적화를 tensorflow내에서 병렬로 처리하므로 load_image_train도 @tf.function으로 선언해야 한다.
#https://www.tensorflow.org/guide/data_performance?hl=ko
#https://www.facebook.com/groups/TensorFlowKR/posts/911745229166536/
train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

#cache() : 데이터를 램에 캐싱한다
# 전처리(load_image_train) -> 캐싱(cache()) -> 데이터 적재 ->  셔플링(shuffle()) -> 배치(batch()) -> 프리페치(prefetch())
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BUFFER_SIZE)

#데이터셋 API가 항상 한 배치가 미리 준비되도록 한다.
#https://doubly8f.netlify.app/%EA%B0%9C%EB%B0%9C/2020/08/19/tf-loading-preprocessing-data/
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_dataset = test.batch(BATCH_SIZE)

# 테스트를 위해서 데이터셋 내에 존재하는 1개의 Input Image, Ground-Truth를 불러오고 화면에 띄워 확인합니다.
for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])


# MobileNetV2를 이용해서 Down-Sampling 과정을 정의합니다.
base_model = keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]

layers = [base_model.get_layer(name).output for name in layer_names]
down_stack = keras.Model(inputs=base_model.input, outputs=layers)
down_stack.trainable = False

up_stack = [
  upsample(512, 3),
  upsample(256, 3),
  upsample(128, 3),
  upsample(64, 3),
]

class UNET(keras.Model):
  def __init__(self, output_channels):
    super(UNET, self).__init__()
    self.down_stack = down_stack
    self.up_stack = up_stack

    self.last = keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')
  
  def call(self, x):
    skips = self.down_stack(x)
    #마지막 feature를 input으로
    x = skips[-1]
    #reversed로 배열
    skips = reversed(skips[:-1])

    for up, skip in zip(self.up_stack, skips):
      x = up(x)
      concat = keras.layers.Concatenate()
      # x와 skip 두 input을 붙여 하나의 배열로 만들고 다음 up의 input으로 사용
      x = concat([x, skip])
    
    x = self.last(x)

    return x

model = UNET(OUTPUT_CHANNELS)

loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
@tf.function
def sparse_cross_entropy_loss(logits, y):
  return tf.reduce_mean(loss_object(y, logits))

optimizer = tf.optimizers.Adam(learning_rate)

@tf.function
def train_step(model, x, y):
  with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = sparse_cross_entropy_loss(y_pred, y)
  
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

show_predictions()

for epoch in range(EPOCHS):
  train_epoch_loss = 0.0

  for batch_x, batch_y in train_dataset:
    train_step(model, batch_x, batch_y)

    cur_loss = sparse_categorical_crossentropy(model(batch_x), batch_y)
    train_epoch_loss = train_epoch_loss + cur_loss
  
  train_epoch_loss = train_epoch_loss / len(list(train_dataset))

  validation_epoch_loss = 0.0

  for batch_x, batch_y in test_dataset:
    cur_loss = sparse_categorical_crossentropy(model(batch_x), batch_y)
    validation_epoch_loss = validation_epoch_loss + cur_loss
  
  validation_epoch_loss = validation_epoch_loss / len(list(test_dataset))

  print(f'epoch : {epoch+1}, train_loss : {train_epoch_loss}, validation-loss : {validation_epoch_loss}')

show_predictions(test_dataset, 3)