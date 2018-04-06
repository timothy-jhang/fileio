import tensorflow as tf

filename_queue = tf.train.string_input_producer(["../../fer2013/fer2013.csv"])
# it means you choose to skip the first line for every file in the queue
reader = tf.TextLineReader(skip_header_lines=1) # skip the first line in the file
key, value = reader.read(filename_queue)

with tf.Session() as sess:
 coord = tf.train.Coordinator()
 threads = tf.train.start_queue_runners(coord=coord)
 print sess.run(key) # data/heart.csv:2
 print sess.run(value) # 144,0.01,4.41,28.61,Absent,55,28.87,2.06,63,1
 coord.request_stop()
 coord.join(threads)
