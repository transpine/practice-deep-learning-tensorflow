# -*- coding: utf-8 -*-

import scipy.misc as misc
import os, sys
import tarfile
import zipfile
from six.moves import urllib

def save_image(image, save_dir, name, mean=None):
  """
  만약 평균값을 argument로 받으면 평균값을 더한뒤에 이미지를 저장하고, 아니면 바로 이미지를 저장합니다.
  """
  if mean:
    image = unprocess_image(image, mean)
  misc.imsave(os.path.join(save_dir, name + ".png"), image)

# 이미지에 평균을 더합니다.
def unprocess_image(image, mean_pixel):
  return image + mean_pixel

# dir_path에 url_name에서 다운받은 zip파일의 압축을 해제합니다.
def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  filename = url_name.split('/')[-1]
  filepath = os.path.join(dir_path, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write(
          '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    if is_tarfile:
      tarfile.open(filepath, 'r:gz').extractall(dir_path)
    elif is_zipfile:
      with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dir_path)