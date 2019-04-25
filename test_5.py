import numpy as np
import re
import tensorflow as tf
import pickle
import collections
import random

np.random.seed(16)
data = {}
data['input'] = []
data['response'] = []

with open('input.txt', 'r') as f:
	for line in f:
		data['input'].append(line)

with open('response.txt', 'r') as f:
	for line in f:
		data['response'].append(line)

# training_data = []
dump = []

for i in range(len(data['input'])):
	dump.append(data['input'][i])
	dump.append(data['response'][i])
	# training_data.append([data['input'][i], data['response'][i]])

logs_path = 'output/rnn_words'
writer = tf.summary.FileWriter(logs_path)

def make_data(data):
    content = [x.strip() for x in data]
    content = np.array(content)
    content = np.reshape(content, [-1,])
    return content

training_data2 = make_data(dump)

def make_dictionary(words):
	count = collections.Counter(words).most_common()
	data['diction'] = {}
	for word, _ in count:
		data['diction'][word] = len(data['diction'])
	data['rev_diction'] = dict(zip(data['diction'].values(), data['diction'].keys()))


make_dictionary(training_data2)

vocab_size = len(data['diction'])

vocab_size = len(data['diction'])
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_hidden = 512

x = tf.placeholder(tf.float32, [None,1,1])
y = tf.placeholder(tf.float32, [None,vocab_size])

weights = { 'out' : tf.Variable(tf.random_normal([n_hidden,vocab_size])) }
biases = { 'out' : tf.Variable(tf.random_normal([vocab_size]))}

layer_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden)
layer_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden)
rnn_cell = tf.contrib.rnn.MultiRNNCell([layer_1, layer_2])

outputs, states, _ = tf.contrib.rnn.static_rnn(rnn_cell,x,tf.unstack(tf.transpose(x, perm=[1, 0, 2])) ,dtype=tf.float32)

pred = tf.add(tf.matmul(outputs[-1], weights['outs']), biases['out'])


cost = tf.reduce_mean(tf.nn.softmax_cross_entropywith_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_intializer)
	acc_total = 0
	loss_total = 0
	num = random.randint(0, 2)
	while step < training_iters:
		if num > (len(data['dump']) - 2):
			num = random.randint(0, 2)
			x_data =[[data['diction'][j] for i in data['dump'][num:(num + 2)] ]]
			x_data = np.reshape(np.array(x_data), [-1,1,1])
			y_data = np.zeros([vocab_size], dtype=float)
			tmp = [str(j) for j in [x.strip() for x in re.split('(\W+)?', s2) if x.strip()]]
			for i in tmp:
				y_data[data['diction'][i]] = 1.0
			y_data = np.reshape(y_data, [1,-1])
			_, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, pred], feed_dict={x:x_data, y:y_data})
			loss_total += loss
			acc_total += acc
			if (step+1) % display_step == 0:
				print("Iter=" + str(step+1) + ",Average Loss = {:.6f}".format(loss_total/display_step) + ", Average Accuracy= {:.2f}%".format(100*acc_total/display_step))
				acc_total = 0
				loss_total = 0
		step += 1
	saver.save(sess, 'output/model_5/model')
pickle.dump( data, open( "output/model_5/training_data", "wb" ) )