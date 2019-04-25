import numpy as np
import tensorflow as tf 
import re 
from tensorflow.python.ops import math_ops

logs_path = 'output/rnn_words'
writer = tf.summary.FileWriter(logs_path)

training_file = 'data.txt'

data = {}

def read_data(fname):
	content = []
	words = []
	with open(fname,'r') as f:
		for line in f:
			a = line.strip()
			b = [x.strip() for x in re.split('(\W+)?',a) if x.strip()]
			content.append(b)
			for i in b:
				words.append(i)
	content = np.array(content)
	content = np.reshape(content, [-1,])
	data['dump'] = content
	data['words'] = list(set(words))

read_data(training_file)


def build_diction(words):
	data['diction'] = dict((v,i) for i,v in enumerate(words))
	data['rev_diction'] = dict(zip(data['diction'].values(), data['diction'].keys()))

build_diction(data['words'])
vocab_size = len(data['diction'])


def batch_gen():
	for i in range(len(data['dump'])-1):
		yield data['dump'][i], data['dump'][i+1]


learning_rate = 0.001
training_itrs = 10
display_step = 1
n_input = 1
n_hidden = 512

x_place = tf.placeholder("float", [None,])
y_place = tf.placeholder("float", [None,])

weights = { 'out' : tf.Variable(tf.random_normal([None,n_hidden,vocab_size])) }
biases = { 'out' : tf.Variable(tf.random_normal([vocab_size])) }


# x_place = tf.reshape(x_place, [-1, n_input])
# x_place = tf.split(x_place, n_input,1)
x_ = tf.reshape(x_place, [-1,1,1])

layer_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, activation=math_ops.sigmoid)
layer_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, activation=math_ops.sigmoid)
rnn_cell = tf.contrib.rnn.MultiRNNCell([layer_1, layer_2])

# output, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
output, states = tf.nn.dynamic_rnn(rnn_cell,x_,dtype=tf.float32)

pred = tf.add(tf.matmul(output, weights['out']),biases['out'])


oh = tf.one_hot(tf.cast(y_place, tf.int32), vocab_size, on_value=1.0,axis=-1, dtype=tf.float32)
y_ = tf.reshape(oh, [1, tf.size(y_place),vocab_size])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y_place,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	step = 0
	acc_total = 0
	loss_total = 0
	writer.add_graph(sess.graph)
	while step < training_itrs:
		for i in batch_gen():
			x_data = np.array([data['diction'][str(a)] for a in i[0]])
			# x_data = np.reshape(np.array(x_data), [-1,n_input,1])
			# y_1 = []
			# for x in i[1]:
			# 	y = np.zeros([vocab_size], dtype=float)
			# 	y[data['diction'][str(x)]] = 1
			# 	y_1.append(y)
			# y_data = np.array(y_1)
			y_data = np.array([data['diction'][str(a)] for a in i[1]])
			# l_y = len(y) 
			# oh = tf.one_hot(y, vocab_size, on_value=1.0, axis=-1)
			# y_data = tf.reshape(oh, [1,l_y,vocab_size])
			_, acc, loss, one_hot_pred = sess.run([optimizer, accuracy, cost, pred], feed_dict={x_place:x_data, y_place:y_data})
			loss_total += loss
			acc_total += acc
			if (step + 1 ) % display_step == 0:
				print("Iter= " + str(step+1) + ", Average Loss= " + \
	                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
	                  "{:.2f}%".format(100*acc_total/display_step))
				acc_total = 0
				loss_total = 0
				rep_word = [data['rev_diction'][np.argmax(x)] for x in one_hot_pred]
				# rep_word = data['rev_diction'][int(tf.argmax())]
				print("input: %s" , i[0])
				print("response: %s", i[1])
				print("predicted response: %s", rep_word)
		step += 1