# -*- coding: utf-8 -*-
 
import tensorflow
import os
import tensorflow.keras as keras
import json
from sklearn.utils import shuffle
from devfacPyLogger import log
from utils import load_image
from tqdm import tqdm
import numpy as np

 # Caption annotation 압축파일을 다운받고, annotations 폴더에 압축을 풉니다.
def maybe_download_and_extract():
    # annotation_dir = '/annotations/'
    annotation_dir = '\\annotations\\'
    
    if not os.path.exists(os.path.abspath('.') + annotation_dir):
      # cache_subdir을 주어야 현재 위치에 압축이 풀린다.
      # annotations_trainval2014.zip안에 annotations 디렉토리로 압축이 되어 있으므로 '.'만 cache_subdir로 준다.
      annotation_zip = keras.utils.get_file('captions.zip', 
                                            cache_subdir=os.path.abspath('.'),
                                            origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                            extract = True)
      annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
      os.remove(annotation_zip)
    else:
      annotation_file = os.path.abspath('.') + annotation_dir + 'captions_train2014.json'

    # image_folder = '/train2014/'
    image_folder = '\\train2014\\'

    if not os.path.exists(os.path.abspath('.') + image_folder):
      image_zip = keras.utils.get_file('train2014.zip',
                                        cache_subdir=os.path.abspath('.'),
                                        origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                        extract = True)
      PATH = os.path.dirname(image_zip) + image_folder
      os.remove(image_zip)
    else:
      PATH = os.path.abspath('.') + image_folder

    return annotation_file, PATH

def prepare_image_and_caption_data(num_examples=30000):
  annotation_file, PATH = maybe_download_and_extract()

  with open(annotation_file, 'r') as f:
    annotations = json.load(f)

  all_captions = []
  all_img_name_vector = []

  for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end> '
    image_id = annot['image_id']
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % ( image_id )

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)
  

  train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

  train_captions = train_captions[:num_examples]
  img_name_vector = img_name_vector[:num_examples]

  log.debug(f'selected sampled : {len(train_captions)}')
  log.debug(f'all samples : {len(all_captions)}')

  print(train_captions[0])
  print(img_name_vector[0])
  return train_captions, img_name_vector

def cache_bottlenecks(img_name_vector, image_features_extract_model):
  encode_train = sorted(set(img_name_vector))

  image_dataset = tensorflow.data.Dataset.from_tensor_slices(encode_train)
  image_dataset = image_dataset.map(load_image, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE).batch(16)

  # 동일 이미지에 대한 feature map 변환 연산을 반복수행하는 부분을 제거하기 위해서
  # 한번 feature map 형태로 변환한 값들을 disk에 저장해서 caching합니다.
  for img, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(img)

    # 16x8x8x2048 이미지를 16x64x2048 형태로 reshape합니다.
    batch_features = tensorflow.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

    for bf, p in zip(batch_features, path):
      path_of_feature = p.numpy().decode('utf-8')
      # log.debug(f'p: {p}')
      np.save(path_of_feature, bf.numpy())
