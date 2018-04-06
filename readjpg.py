import tensorflow as tf

filename_queue = tf.train.string_input_producer(["./face2.jpg"])
# it means you choose to skip the first line for every file in the queue
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

image = tf.image.decode_jpeg(value,channels=3)
print('>>> reshape_image = ', image)

with tf.Session() as sess:
 coord = tf.train.Coordinator()
 threads = tf.train.start_queue_runners(coord=coord)
 print sess.run(key) # data/heart.csv:2
 print sess.run(image) # 144,0.01,4.41,28.61,Absent,55,28.87,2.06,63,1
 coord.request_stop()
 coord.join(threads)
