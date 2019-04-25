import numpy as np 
import collections
import random
import tensorflow as tf 

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

learning_rate = 0.001
epochs = 50000
d_steps = 1000
n_input = 2
n_hidden = 512

x = tf.placeholder("float", [None, n_input,1])
y = tf.placeholder("float", [None, vocab_size])

weights = { 'out' : tf.Variable(tf.random_normal([n_hidden, vocab_size])) }
bias = { 'out': tf.Variable(tf.random_normal([vocab_size])) }

def rnn(x, weights, bias):
	x = tf.reshape(x, [-1, n_input])
	x = tf.split(x, n_input,1)
	rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden), tf.contrib.rnn.BasicLSTMCell(n_hidden)])
	outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
	return tf.add(tf.matmul(outputs[-1], weights['out']), bias['out'])

pred = rnn(x, weights, bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	step = 0
	offset = random.randint(0, n_input+1)
	end_offset = n_input + 1
	acc_total = 0
	loss_total = 0

	writer.add_graph(sess.graph)

	while step < epochs:
		if offset > (len(training_data2)-end_offset):
			offset = random.randint(0, n_input+1)

		symbols_in_keys = [[data['diction'][str(training_data2[i])]] for i in range(offset, offset+n_input)]
		symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input,1])

		symbols_out_onehot = np.zeros([vocab_size], dtype=float)
		symbols_out_onehot[data['diction'][str(training_data2[offset+n_input])]] = 1.0
		symbols_out_onehot = np.reshape(symbols_out_onehot, [1,-1])

		_, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, pred], feed_dict={x:symbols_in_keys, y:symbols_out_onehot})
		loss_total += loss 
		acc_total += acc
		if (step+1) % d_steps == 0:
			print("Iter=" + str(step+1) + ",Average Loss=" + \
				"{:.6f}".format(loss_total/d_steps) + ", Average Accuracy =" + \
				"{:.2f}%".format(100*acc_total/d_steps))
			acc_total = 0
			loss_total = 0
			symbols_in = [training_data2[i] for i in range(offset, offset+ n_input)]
			symbols_out = training_data2[offset + n_input]
			symbols_out_pred = data['rev_diction'][int(tf.argmax(onehot_pred,1).eval())]
			print("%s - [%s] vs [%s]" %(symbols_in, symbols_out, symbols_out_pred))
		step +=1
		offset += (n_input+1)
	saver.save(sess, 'output/model_1')

