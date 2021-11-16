import tensorflow as tf
import os
import keras
def download_and_extract():
    annotation_dir = '/annotations/'
    
    if not os.path.exists(os.path.abspath('.') + annotation_dir):
      annotation_zip = tf.keras.utils.get_file_zip('captions.zip')