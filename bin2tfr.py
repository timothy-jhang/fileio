# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read CIFAR10 style bin file and writes TFRecords.

Generates tf.train.Example protos and writes them to TFRecord files from bin files
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path
import glob
from struct import *

import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _get_file_names(data_dir):
  """Returns the file names expected to exist in the input_dir."""
  
# gender data 
  file_names = {}
  print('>> data_dir = ', data_dir)
  file_names['train'] = glob.glob(data_dir + 'gender_train/256*.bin')
  file_names['valdation'] = glob.glob(data_dir + 'gender_val/256*.bin')
  file_names['eval'] = glob.glob(data_dir + 'gender_test/256*.bin')
  print('file_names = ', file_names)
  return file_names


#
#bin file to tfrecord conversion KSJHANG, 2018.04.27
def convert_to_tfrecord(input_files, output_file):
  """Converts a file to TFRecords."""
  print('>> Generating %s' % output_file)
  label_bytes = 1
  height = 256
  width = 256
  depth = 3
  image_bytes = height*width*depth
  record_bytes = label_bytes + image_bytes
  print('>> record bytes = ', record_bytes)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      print('>> input_file = ', input_file)
      fs = os.path.getsize(input_file)
      print('>> file=',input_file,'size=', fs) 
      num_entries_in_batch = fs // record_bytes
      print('>> num entries = ', num_entries_in_batch)
      with open(input_file, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        print('fileContent type=', len(fileContent))

        for i in range(num_entries_in_batch):
          img=fileContent[record_bytes*i+1:record_bytes*(i+1)]
          p = unpack('B',fileContent[record_bytes*i]) 
   #       print('p=', p, type(p))
          label= p[0]
          example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(img),
                'label': _int64_feature(label)
            }))
          record_writer.write(example.SerializeToString())


def main(data_dir):
#bin file to tfrecords conversion
  print('> top folder = ', data_dir)
  file_names = _get_file_names(data_dir)

  # tfr_dir  in case of gender data  
  tfr_dir = data_dir + 'gender/'
  print('> tfr_dir = ', tfr_dir)
  if not os.path.isdir(tfr_dir):
    os.mkdir(tfr_dir, 0755 )

  for mode, files in file_names.items():
    print('> mode=',mode)
    input_files = files

    output_file = os.path.join(tfr_dir, mode + '.tfrecords')
    try:
      os.remove(output_file)
    except OSError:
      pass
    # Convert to tf.train.Example and write the to TFRecords.
    convert_to_tfrecord(input_files, output_file)
  print('Done!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      default='',
      help='Directory to download and extract CIFAR-10 to.')

  args = parser.parse_args()
  main(args.data_dir)

