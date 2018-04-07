# read csv and write into cifar10 bin file
import tensorflow as tf
import numpy as np

DATA_PATH = './fer2013.csv'

def read_csv():
  filename_queue = tf.train.string_input_producer([DATA_PATH])
# it means you choose to skip the first line for every file in the queue
  reader = tf.TextLineReader(skip_header_lines=1) # skip the first line in the file
  _, value = reader.read(filename_queue)

# original rec_def
  rec_def=[ [1], [''], [''] ]
# splitted rec_def - splitted file has not final field('Training', 'PubicTest', 'PrivateTest')
#  rec_def=[ [1], [''] ]
  content = tf.decode_csv(value, record_defaults=rec_def,field_delim=',')
  #print('>>> content =', len(content) )
  data = content[1]
  label = content[0]
  return label, data

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
  with tf.Session() as sess:
    label,data = read_csv()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

# Train file (the first part)
    num_bytes=0
    for i in range(NUM_TRAINS+1):
      d,l = sess.run([data, label])
      ia = str(l) + ' ' + d 
      ia = [ np.uint8(int(f)) for f in ia.split() ] 
      ba= bytearray(ia)
      train_file.write(ba)
      num_bytes = num_bytes + len(ba)
    print('train_file num_bytes = ', num_bytes)

# public test file (the 2nd part)
    num_bytes=0
    for i in range(NUM_PUBTST+1):
      d,l = sess.run([data, label])
      ia = str(l) + ' ' + d 
      ia = [ np.uint8(int(f)) for f in ia.split() ] 
      ba= bytearray(ia)
      pubtst_file.write(ba)
      num_bytes = num_bytes + len(ba)
    print('pub test file num_bytes = ', num_bytes)

# private test file (the 3rd part)
    num_bytes=0
    for i in range(NUM_PRITST+1):
      d,l = sess.run([data, label])
      ia = str(l) + ' ' + d 
      ia = [ np.uint8(int(f)) for f in ia.split() ] 
      ba= bytearray(ia)
      pritst_file.write(ba)
      num_bytes = num_bytes + len(ba)
    print('priv test file num_bytes = ', num_bytes)
    
    coord.request_stop()
    coord.join(threads)
    train_file.close()
    pubtst_file.close()
    pritst_file.close()

if __name__ == '__main__':
  main()


