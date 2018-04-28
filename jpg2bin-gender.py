import tensorflow as tf
import os
import numpy as np

def readjpg(filename_queue):
# it means you choose to skip the first line for every file in the queue
  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)

  image = tf.image.decode_jpeg(value,channels=3)
  # make it 48x48x3 for testing 
  # imager = tf.image.resize_images(image, [48,48] )
  imager = tf.image.resize_images(image, [256,256] )
  #jpg image : 48x48x3, float, each btw 0.-255. 
  # flattening channel by channel 
  #r1 = tf.reshape(imager[:,:,0],[48*48])
  #r2 = tf.reshape(imager[:,:,1],[48*48])
  #r3 = tf.reshape(imager[:,:,2],[48*48])
  r1 = tf.reshape(imager[:,:,0],[256*256])
  r2 = tf.reshape(imager[:,:,1],[256*256])
  r3 = tf.reshape(imager[:,:,2],[256*256])
  print('rx shape = ', r3.shape)
  imager = tf.concat([r1,r2,r2],0)
  return imager

def main(data_dir, i):

  with tf.Session() as sess:
     print('label=', i)
     src_dir = data_dir+str(i)+'/'
     filenames = [ os.path.join(src_dir,f) for f in os.listdir(src_dir) ] 
     print('>> # of files in ',src_dir,'=', len(filenames))
     no_files = len(filenames)
# file existence test
#     for f in filenames:
#       if not tf.gfile.Exists(f):
#         raise ValueError('Failed to find file: ' + f)
     filename_queue = tf.train.string_input_producer(filenames)
     image = readjpg(filename_queue)
     #train_file = open(data_dir+'48x48x3_'+str(i)+'.bin', "wb")
     target_file = data_dir+'_256x256x3_'+str(i)+'.bin'
     train_file = open(target_file, "wb")
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
       print('> j=',j)
       train_file.write(ba)
     train_file.close()
     coord.request_stop()
     coord.join(threads)


if __name__ == '__main__':
# gender: assumed folder structure  '../fold0/gender_train,test,val/[0~1]/*.jpg'
# age:  assumed folder structure  '../fold0/age_train,test,val/[0~7]/*.jpg'
  for top_fold in ['../fold1/', '../fold2/', '../fold3/', '../fold4/']:
#gender
#   data_dirs = ['gender_test/', 'gender_train/', 'gender_val/']
# age
    data_dirs = ['age_test/', 'age_train/', 'age_val/']
    for data_dir in data_dirs: 
      fdir = top_fold + data_dir
      print('data_dir = ', fdir)
#gender
#      for i in range(2): # 2 classes - gender classification
# age
      for i in range(8): # 8 classes - age classification
        main(fdir, i)


