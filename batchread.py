import tensorflow as tf


BATCH_SIZE = 10
N_FEATURES = 1
# min # of elements after dequeue
min_after_dequeue = 10 * BATCH_SIZE

#max # elements in queue
capacity = 20 * BATCH_SIZE
DATA_PATH = '../../fer2013/fer2013.csv'
def batch_gen(filenames):
  filename_queue = tf.train.string_input_producer(filenames)

# it means you choose to skip the first line for every file in the queue
  reader = tf.TextLineReader(skip_header_lines=1) # skip the first line in the file
  _, value = reader.read(filename_queue)
  print('>>> value=', value)

  rec_def=[ [1], [''], [''] ]
  content = tf.decode_csv(value, record_defaults=rec_def,field_delim=',')
  print('>>> content =', content)

  data = content[0]
  label = content[1]
  flag  = content[-1]

  labelbatch, databatch, flagbatch = tf.train.shuffle_batch( [label, data, flag], batch_size = BATCH_SIZE, capacity=capacity, 
		min_after_dequeue = min_after_dequeue)
  print(labelbatch)
  print(databatch)
  print(flagbatch)
  return labelbatch, databatch, flagbatch

with tf.Session() as sess:
#the ordering is important:batch_gen should proceed coordinator
 db, lb, fb = batch_gen([DATA_PATH])

 coord = tf.train.Coordinator()
 threads = tf.train.start_queue_runners(coord=coord)
 for i in range(2):
   lb0, db0, fb0 = sess.run([lb, db, fb ] ) 
   print('lb=', lb0)
   print('db=', db0)
   print('fb=', fb0)
 coord.request_stop()
 coord.join(threads)

