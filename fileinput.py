#tensorflow reading jpg data 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

#from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def inputs(data_dir,label):
  with tf.device('/cpu:0') : 
    path = data_dir+str(label)+'/'
    print('path=',path)
    print('label=',label)
    filenames = [ os.path.join(path,f) for f in os.listdir(path) ] 
    print('>>>>>>>>> len(filenams)=',len(filenames))
    for f in filenames:
      print('f=',f)
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('input'):
      # Create a queue that produces the filenames to read.
      filename_queue = tf.train.string_input_producer(filenames[0:2])

      image_reader = tf.WholeFileReader()
      _, image_file = image_reader.read(filename_queue)

      reshaped_image = tf.image.decode_jpeg(image_file,channels=3)
      print('>>> reshape_image = ', reshaped_image)
    
    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
      height = 32
      width = 32
      resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
      print('>>> resized image = ', resized_image)

    # Subtract off the mean and divide by the variance of the pixels.
      float_image = tf.image.per_image_standardization(resized_image)
      print('>>> float_image =', float_image)
   
  return float_image

def main(argv=None):  # pylint: disable=unused-argument
  float_image = inputs('/home/sun/cae/fold0/age_test/', str(0) )

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(8):
	fi = sess.run(float_image)
	print('fi = ', fi.shape)

  # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  tf.app.run()

