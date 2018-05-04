import tensorflow as tf
import os
import numpy as np
from skimage import io
import dlib
import glob
#import matplotlib.pyplot as plt
from skimage.transform import resize

def main(data_dir, i ):
  print('label=', i)
  src_dir = data_dir+str(i)+'/'
  filenames = glob.glob(src_dir + '/*.jpg') 
  print('files[0] = ', filenames[0])
  print('>> # of files in ',src_dir,'=', len(filenames))
  no_files = len(filenames)
  target_file = data_dir+'__256x256x3_'+str(i)+'.bin'
  train_file = open(target_file, "wb")
  for fname in filenames:
#    read image, convert image and convert into bin
    image = io.imread(fname)
    image = resize(image, (256,256))
    print('image = ',type(image), image.shape)

    # label + image(256x256) 
    label = np.uint8(i)
    ia = [ np.uint8(f) for f in image ]
    ia = np.insert(ia,0, label)
    ba= bytearray(ia)
    train_file.write(ba)
  train_file.close()


if __name__ == '__main__':
  #detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat.bz2')
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# gender: assumed folder structure  '../fold0/gender_train,test,val/[0~1]/*.jpg'
# age:  assumed folder structure  '../fold0/age_train,test,val/[0~7]/*.jpg'
  for top_fold in ['../fold0/', '../fold1/', '../fold2/', '../fold3/', '../fold4/']:
#gender
# age
#   data_dirs = ['age_test/', 'age_train/', 'age_val/']
    data_dirs = ['gender_val/', 'gender_test/', 'gender_train/']
    for data_dir in data_dirs: 
      fdir = top_fold + data_dir
      print('data_dir = ', fdir)
#gender : 2 classes
# age : 8 classes
#      for i in range(8): # 8 classes - age classification
      for i in range(2): # 2 classes - gender classification
        main(fdir, i)



