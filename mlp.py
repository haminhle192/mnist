import tensorflow as tf
import numpy as np
import math

IMAGE_SIZE = 20
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_LABEL = 10
FILE_NAME = "data.csv"
ERROR_FILE_NAME = "errorData.csv"

def exportWeight(W, b):
       
    return

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def convertData(digit):
    one_hot = np.zeros([10])
    one_hot[int(digit)] = 1
    
    return one_hot

def getData(filename):
    file_name_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader()
    key , value = reader.read(file_name_queue) #key is file name & value is data inside
    record_defaults = np.zeros([401, 1]).tolist() #pattern of file
    data = tf.decode_csv(value, record_defaults = record_defaults)
    features = tf.stack(data[:400])
    labels = tf.stack(data[400])
    
    x = np.array([])
    y = np.array([])
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        
        for i in range(file_len(filename)):
            tx, ty = sess.run([features, labels])
            if ty == -1:
                continue
            ty = convertData(ty)
            if len(x) == 0:
                x = np.array([tx])
                y = np.array([ty])
 #               x = np.vstack((x, tx))
#                y = np.vstack((y, ty))
#                x = np.delete(x,0,0)
#                y = np.delete(y,0,0)
            else:
                x = np.vstack((x, tx))
                y = np.vstack((y, ty))
    
        coord.request_stop()
        coord.join(threads)

    return (x, y)

def buildModel(images, hidden1_units):
    
    #Build Hidden layer 1
    with tf.name_scope('hidden_1'):
        #weights for input layer
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev = 1.0/float(IMAGE_SIZE)), name = 'weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name = 'biases')
        hidden1 = tf.matmul(images, weights) + biases
    
    #Build Output layer
    with tf.name_scope('softmax_linear'):
        #weight for hidden layer 1
        weights = tf.Variable(
                              tf.truncated_normal([hidden1_units, NUM_LABEL],
                                                  stddev = 1.0 / math.sqrt(float(hidden1_units))),
                              name='weights')
        biases = tf.Variable(tf.zeros([NUM_LABEL]),
                         name='biases')
        logits = tf.matmul(images, weights) + biases
    
    return logits

def loss():
    
    return

def test(features, labels):
    sess = tf.Session()
    
    x = tf.placeholder(tf.float32, [None, 400], name = "x")
    y = tf.placeholder(tf.float32, [None, 10],name = "y")
    W = tf.Variable(tf.zeros([400, 10]), tf.float32, name = "W")
    b = tf.Variable(tf.zeros([10]), tf.float32, name = "b")
 #   W = tf.Variable(..., tf.float32, name = "W")
 #   b = tf.Variable(..., tf.float32, name = "b")

    saver = tf.train.Saver()
    
    linear_model = tf.matmul(x,W) + b
    softmax = tf.nn.softmax(linear_model)

    sess.run(tf.global_variables_initializer())
    
#    mean_square = tf.square(linear_model - y)
    cross_entrophy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = linear_model)

    loss = tf.reduce_sum(cross_entrophy)
    
    optimizer = tf.train.GradientDescentOptimizer(0.0005)
    train = optimizer.minimize(loss)
    saver.restore(sess, "demoVariable/softmax_model.ckpt")

    if False:
           for i in range(20000):
               _, curloss = sess.run([train, loss], feed_dict = {x : features, y : labels})
               if i%1000 == 0:
                   print("LOSS: ", i, curloss);

           print(sess.run(b))

    correct_prediction = tf.equal(tf.argmax(softmax,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
  #  features, labels = getData(ERROR_FILE_NAME)

    result = sess.run([tf.argmax(softmax,1), tf.argmax(y,1)], feed_dict={x: features, y: labels})
 #   print(result)
    print(sess.run(accuracy, feed_dict = {x:features, y: labels}))
    for i in range(len(labels)):
           print(result[0][i], result[1][i])
 #   print(sess.run(b))

    save_path = saver.save(sess, "demoVariable/softmax_model.ckpt")

    tf.summary.FileWriter("demoVariable", sess.graph)

    return 

if __name__ == '__main__':
#    x , y = getData()
    features,labels = getData(FILE_NAME)
    test(features, labels)
    
