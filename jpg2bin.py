import tensorflow as tf
import os
import numpy as np

def readjpg(filename_queue):
# it means you choose to skip the first line for every file in the queue
  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)

  image = tf.image.decode_jpeg(value,channels=3)
  # make it 48x48x3 for testing 
  imager = tf.image.resize_images(image, [48,48] )
  #jpg image : 48x48x3, float, each btw 0.-255. 
  # flattening channel by channel 
  r1 = tf.reshape(imager[:,:,0],[48*48])
  r2 = tf.reshape(imager[:,:,1],[48*48])
  r3 = tf.reshape(imager[:,:,2],[48*48])
  print('rx shape = ', r3.shape)
  imager = tf.concat([r1,r2,r2],0)
  return imager

def main(data_dir, i):

  with tf.Session() as sess:
     print('label=', i)
     filenames = [ os.path.join(data_dir+str(i)+'/',f) for f in os.listdir(data_dir+str(i)+'/') ] 
     print('>>>>>>>>> len(filenames)=',len(filenames))
     no_files = len(filenames)
# file existence test
#     for f in filenames:
#       if not tf.gfile.Exists(f):
#         raise ValueError('Failed to find file: ' + f)
     filename_queue = tf.train.string_input_producer(filenames)
     image = readjpg(filename_queue)
     train_file = open(data_dir+'48x48x3_'+str(i)+'.bin', "wb")
     coord = tf.train.Coordinator()
     threads = tf.train.start_queue_runners(coord=coord)
     print('label=',i,no_files,'files')
     for j in range(no_files):
       img =  sess.run([image]) 
       #ia = str(i) + img
       label = np.uint8(i)
       ia = [ np.uint8(f) for f in img[0] ]
       ia = np.insert(ia,0, label)
       ba= bytearray(ia)
       train_file.write(ba)
     train_file.close()
     coord.request_stop()
     coord.join(threads)


if __name__ == '__main__':
# assumed folder structure  '../cae/fold0/age_test/[0-7]/*.jpg'
  data_dir = '../cae/fold0/age_test/'
  for i in range(8): # 8 classes 0...7
    main(data_dir, i)

