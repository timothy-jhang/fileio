# read csv and write into cifar10 bin file
import numpy as np
import csv
import os
import cv2

DATA_PATH = './fer2013.csv'
NUM_TRAINS=28709
NUM_PUBTST=3589
NUM_PRITST=3589
TRAIN_DIR='./train/'
EVAL_DIR='./eval/'
LABELS=[0,1,2,3,4,5,6]
def main( ):
# prepare folders 
  if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)
  for label in LABELS:
    fold = os.path.join(TRAIN_DIR, str(label))
    print('fold=',fold)
    if not os.path.exists(fold):
      os.mkdir(fold)
  if not os.path.exists(EVAL_DIR):
    os.mkdir(EVAL_DIR)
  for label in LABELS:
    fold = os.path.join(EVAL_DIR, str(label))
    print('fold=',fold)
    if not os.path.exists(fold):
      os.mkdir(fold)

  with open(DATA_PATH, 'r') as csvfile:
    csvread = csv.DictReader(csvfile, delimiter=',')
    for i, row in enumerate(csvread):
#emotion,pixels,Usage
#########print(row['emotion'], row['pixels'], row['Usage'])
      ia=row['pixels']; l = row['emotion']
      ia = [ np.uint8(int(f)) for f in ia.split() ] 
      ia = np.asarray(ia)
      ia = np.reshape(ia, (48,48))
      ia = cv2.cvtColor(ia,cv2.COLOR_GRAY2RGB)
#      ba= bytearray(ia)
      if row['Usage']=='Training': #i < NUM_TRAINS :
        fpath = os.path.join(TRAIN_DIR, str(l)+'/'+str(i)+'.png')
        print('fpath=', fpath)
        cv2.imwrite(fpath, ia)
      elif row['Usage']=='PublicTest': 
        fpath = os.path.join(EVAL_DIR, str(l)+'/'+str(i)+'.png')
        print('fpath=', fpath)
        cv2.imwrite(fpath, ia)
      else:
        fpath = os.path.join(EVAL_DIR, str(l)+'/'+str(i)+'.png')
        print('fpath=', fpath)
        cv2.imwrite(fpath, ia)

if __name__ == '__main__':
  main()


