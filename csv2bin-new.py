# read csv and write into cifar10 bin file
import numpy as np
import csv

DATA_PATH = './fer2013.csv'
NUM_TRAINS=28709
NUM_PUBTST=3589
NUM_PRITST=3589
TRAIN_FILE='fer2013train.bin'
PUBTST_FILE='fer2013pubtst.bin'
PRITST_FILE='fer2013pritst.bin'

def main( ):
  train_file = open(TRAIN_FILE, "wb")
  pubtst_file = open(PUBTST_FILE, "wb")
  pritst_file = open(PRITST_FILE, "wb")
  with open(DATA_PATH, 'rb') as csvfile:
    csvread = csv.DictReader(csvfile, delimiter=',')
    for i, row in enumerate(csvread):
#emotion,pixels,Usage
#########print(row['emotion'], row['pixels'], row['Usage'])
      d=row['pixels']; l = row['emotion']
      ia = str(l) + ' ' + d 
      ia = [ np.uint8(int(f)) for f in ia.split() ] 
      ba= bytearray(ia)
      if row['Usage']=='Training': #i < NUM_TRAINS :
        train_file.write(ba)
      elif row['Usage']=='PublicTest': 
        pubtst_file.write(ba)
      else:
        pritst_file.write(ba)

    train_file.close()
    pubtst_file.close()
    pritst_file.close()

if __name__ == '__main__':
  main()


