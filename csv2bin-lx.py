# read csv and write into cifar10 bin file
import numpy as np
import csv
import cv2
import dlib
from skimage import io
import matplotlib.pyplot as plt
from struct import *

DATA_PATH = './fer2013.csv'
NUM_TRAINS=28709
NUM_PUBTST=3589
NUM_PRITST=3589
TRAIN_FILE='fer2013train.bin'
PUBTST_FILE='fer2013pubtst.bin'
PRITST_FILE='fer2013pritst.bin'

def get_landmarks(image):
  #print('image=',type(image))
  pixels_array = np.asarray(image)
  image = pixels_array.reshape(48, 48) 

  #print('image=',type(image))
  detections = detector(image, 1)
  no_det = len(detections)
  if no_det == 0:
    # no face, then give whole image as a search area for landmark
    dr = dlib.rectangle(1,1,48,48)
    detections = dlib.rectangles()
    detections.append(dr)

  for k, d in enumerate(detections):
    shape = predictor(image, d)
    assert shape.num_parts is 68, "shape.num_parts is  %r" % shape.num_parts
    #if no_det == 0:
    #   print('no_det=0, shape.num_parts=',shape.num_parts)
#    print('shape = ', shape, type(shape))
    xlist = []
    ylist = []
    for i in range(0, 68):
      xlist.append(float(shape.part(i).x))
      ylist.append(float(shape.part(i).y))
    xorg = xlist[27]
    yorg = ylist[27]
    xcentral = [(x - xorg) for x in xlist]
    ycentral = [(y - yorg) for y in ylist]
    landmarks_vectorised = []
    for x, y in zip(xcentral, ycentral):
      landmarks_vectorised.append(x)
      landmarks_vectorised.append(y)

    for i in range(0, 8):
      landmarks_vectorised.append(0)

    # self.data['landmarks_vectorised'] = landmarks_vectorised
  return landmarks_vectorised

#########print(row['emotion'], row['pixels'], row['Usage'])
#def get_avg( ):
#  landmark_avg = np.asarray([0.0] * 144)
#  with open(DATA_PATH, 'rb') as csvfile:
#    csvread = csv.DictReader(csvfile, delimiter=',')
#    for i, row in enumerate(csvread):
#      ia = row['pixels'] 
#      ia = [ np.uint8(int(f)) for f in ia.split() ] 
#      print('i=',i)
#      lx = get_landmarks(ia)
#      if lx is not False:
#        landmark = np.asarray(lx)
#        landmark_avg = landmark_avg + (landmark - landmark_avg)/(i+1.0)*1.0 
#      else:
#        print('i=',i,' no landmark found')
#   print('landmark_avg = ', landmark_avg)
#  return landmark_avg

def main( ):

  train_file = open(TRAIN_FILE, "wb")
  pubtst_file = open(PUBTST_FILE, "wb")
  pritst_file = open(PRITST_FILE, "wb")
#  landmark_avg = get_avg()
  with open(DATA_PATH, 'rb') as csvfile:
    csvread = csv.DictReader(csvfile, delimiter=',')
    for i, row in enumerate(csvread):
#emotion,pixels,Usage
      d=row['pixels']; l = row['emotion']
      ia = row['pixels'] 
      ia = [ np.uint8(int(f)) for f in ia.split() ] 
      lx = get_landmarks(ia)
#      if lx is False:
#	lx = landmark_avg.tolist()
      ia = str(l) + ' ' + d 
      ia = [ np.uint8(int(f)) for f in ia.split() ] 
      ba= bytearray(ia)
      for lp in lx:
        bp = bytearray(pack("f", lp)) 
        ba = ba + bp
      if row['Usage']=='Training': #i < NUM_TRAINS :
        train_file.write(ba)
      elif row['Usage']=='PublicTest': 
        pubtst_file.write(ba)
      else:
        pritst_file.write(ba)

    train_file.close()
    pubtst_file.close()
    pritst_file.close()
    csvfile.close()

if __name__ == '__main__':
  #detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat.bz2')
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  main()


