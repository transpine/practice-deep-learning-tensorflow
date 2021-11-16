#-*- coding:UTF-8 -*-

import os
import inspect
import random
import contextlib2
import io
import PIL.Image
import hashlib

import tensorflow as tf
import numpy as np

from absl import flags
from absl import app

import logging
from colorlog import ColoredFormatter

from lxml import etree

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import label_map_util
from object_detection.utils import dataset_util

from devfacPyLogger import log

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", '', "데이터셋 위치")
flags.DEFINE_string('output_dir', '', 'TFRecords가 저장될 위치')
flags.DEFINE_string('label_map_path', 'object_detection/data/pet_label_map.pbtxt', 'label map proto 파일 위치')
flags.DEFINE_boolean('faces_only', True, 'True이면 얼굴, 아니면 전체 몸으로 sementation')
flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
                    'segmentation masks. Options are "png" or "numerical".')
flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')


def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    A string of the class name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]

def dict_to_tf_example(data, 
                       mask_path, 
                       label_map_dict, 
                       image_subdirectory,
                       ignore_difficult_instances=False,
                       faces_only=True,
                       mask_type='png'):
  img_path = os.path.join(image_subdirectory, data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  if image.format != 'JPEG':
    raise ValueError('이미지 포멧이 JPEG가 아닙니다.')
  key = hashlib.sha256(image)

  with tf.gfile.GFile(mask_path, 'rb') as fid:
    encoded_mask_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_mask_png)
  mask = PIL.Image.open(encoded_png_io)

  if mask.format != 'PNG':
    raise ValueError('마스크 포멧이 PNG가 아닙니다.')

  '''
  mask에 정의된 값을 찾아 min, max 구하기
  '''
  mask_np = np.asarray(mask)
  nonbackground_indices_x = np.any(mask_np !=2, axis=0)
  nonbackground_indices_y = np.any(mask_np !=2, axis=1)
  nonzero_x_indices = np.where(nonbackground_indices_x)
  nonzero_y_indices = np.where(nonbackground_indices_y)

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  masks = []

  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      if faces_only:
        xmin = float(obj['bndbox']['xmin'])
        ymin = float(obj['bndbox']['ymin'])
        xmax = float(obj['bndbox']['xmax'])
        ymax = float(obj['bndbox']['ymax'])
      else:
        xmin = float(np.min(nonzero_x_indices))
        ymin = float(np.min(nonzero_y_indices))
        xmax = float(np.max(nonzero_x_indices))
        ymax = float(np.max(nonzero_y_indices))
      
      #정규화
      xmins.append(xmin/width)
      ymins.append(ymin/height)
      xmaxs.append(xmax/width)
      ymaxs.append(ymax/height)

      class_name = get_class_name_from_filename(data['filename'])
      classes_text.append(class_name.encode('utf-8'))
      classes.append(label_map_dict[class_name])
      truncated.append(int(obj['truncated']))
      poses.append(obj['poses'].encode('utf-8'))

      if not faces_only:
        mask_remapped = (mask_np != 2).astype(np.uint8)
        masks.append(mask_remapped)
    
  feature_dict = {
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
    'image/object/truncated': dataset_util.int64_list_feature(truncated),
    'image/object/view': dataset_util.bytes_list_feature(poses),
  }

  if not faces_only:
    if mask_type == 'numerical':
      mask_stack = np.stack(masks).astype(np.float32)
      masks_flattened = np.reshape(mask_stack, [-1])
      feature_dict['image/object/mask'] = (
          dataset_util.float_list_feature(masks_flattened.tolist()))
    elif mask_type == 'png':
      encoded_mask_png_list = []
      for mask in masks:
        img = PIL.Image.fromarray(mask)
        output = io.BytesIO()
        img.save(output, format='PNG')
        encoded_mask_png_list.append(output.getvalue())
      feature_dict['image/object/mask'] = (
          dataset_util.bytes_list_feature(encoded_mask_png_list))

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(
  output_filename,
  num_shards,
  label_map_dict,
  annotations_dir,
  image_dir,
  examples,
  faces_only=True,
  mask_type='png'):

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
      tf_record_close_stack, output_filename, num_shards
    )

    for idx, example in enumerate(examples):
      if idx % 100 == 0:
        log.info(f'On image {idx} of {len(examples)}')
      xml_path = os.path.join(annotations_dir, 'xmls', example + '.xml')
      mask_path = os.path.join(annotations_dir, 'trimaps', example + '.png')

      if not os.path.exists(xml_path):
        log.warning(f'{xml_path}를 찾을 수 없습니다. example을 무시합니다.')
        continue

      with tf.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      try:
        tf_example = dict_to_tf_example(
          data,
          mask_path,
          label_map_dict,
          image_dir,
          faces_only=faces_only,
          mask_type=mask_type)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      except ValueError:
        log.warning(f'잘못된 example을 무시합니다: {xml_path}')

def main(argv):
  if not FLAGS.data_dir:
    log.error('data_dir 경로가 설정되지 않았습니다.')
    quit()
  if not FLAGS.output_dir:
    log.error('output_dir 경로가 설정되지 않았습니다.')
    quit()

  if not os.path.exists(FLAGS.data_dir):
    log.error('data_dir이 존재하지 않습니다 : {FLAGS.data_dir}')
    quit()
  if not os.path.exists(FLAGS.output_dir):
    log.error(f'output_dir이 존재하지 않습니다 : {FLAGS.label_map_path}')
    quit()
  if not os.path.exists(FLAGS.label_map_path):
    log.error(f'label_map_path가 존재하지 않습니다 : {FLAGS.label_map_path}')
    quit()

  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  log.info('Pet Dataset 읽는중')

  image_dir = os.path.join(data_dir, 'images')
  annotations_dir = os.path.join(data_dir, 'annotations')
  # trimap은 배경,전경분리
  examples_path = os.path.join(annotations_dir, 'trainval.txt')
  examples_list = dataset_util.read_examples_list(examples_path)

  
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(num_examples * 0.7)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  log.info(f'학습 : {len(train_examples)}, Validation: {len(val_examples)}')


  train_output_path = os.path.join(FLAGS.output_dir, 'pet_faces_train.records')
  val_output_path = os.path.join(FLAGS.output_dir, 'pet_faces_val.records')

  if not FLAGS.faces_only:
    train_output_path = os.path.join(FLAGS.output_dir,
                                  'pets_fullbody_with_masks_train.record')
    val_output_path = os.path.join(FLAGS.output_dir,
                                   'pets_fullbody_with_masks_val.record')

  create_tf_record(
    train_output_path,
    FLAGS.num_shards,
    label_map_dict,
    annotations_dir,
    image_dir,
    train_examples,
    faces_only=FLAGS.faces_only,
    mask_type=FLAGS.mask_type,
  )
  create_tf_record(
    val_output_path,
    FLAGS.num_shards,
    label_map_dict,
    annotations_dir,
    image_dir,
    val_examples,
    faces_only=FLAGS.faces_only,
    mask_type=FLAGS.mask_type)

if __name__ == '__main__':
  app.run(main)