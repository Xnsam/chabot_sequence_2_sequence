import numpy as np
import re
import tensorflow as tf

np.random.seed(16)
tf.set_random_seed(16)

model_path = 'output/'
data = {}
filename = 'data.txt'

def make_dataset(filename):
	data['dump'] = []
	data['words'] = []
	with open(filename, 'r') as f:
		for line in f:
			a = line.strip()
			data['dump'].append(a)
			b = [x.strip() for x in re.split('(\W+)?',a) if x.strip()]
			for i in b:
				data['words'].append(i)
	data['words'] = list(set(data['words']))
	data['diction'] = dict((v,i) for i,v in enumerate(data['words']))
	data['rev_diction'] = dict(zip(data['diction'].values(), data['diction'].keys()))


make_dataset(filename)
data['oh'] = []
for i in range(len(data['dump'])):
	b = [x.strip() for x in re.split('(\W+)?',data['dump'][i]) if x.strip()]
	s1 = [data['diction'][str(b[j])] for j in range(len(b))]
	# l = []
	oh = np.zeros([len(data['diction'])], dtype=float)
	for j in s1:
		oh[j] = 1.0
		# l.append(oh)
	data['oh'].append(np.array(oh))


vocab_size = len(data['diction'])
learning_rate = 0.01
training_iters = 50000
display_step = 100
h1 = 1024
h2 = 256
h3 = 128

x = tf.placeholder(tf.float32, [None,vocab_size])
y = tf.placeholder(tf.float32, [None,vocab_size])

weights = {
	'h1' : tf.Variable(tf.random_normal([vocab_size, h1])),
	'h2' : tf.Variable(tf.random_normal([h1, h2])),
	'h3' : tf.Variable(tf.random_normal([h2,h3])),
	'out' : tf.Variable(tf.random_normal([h3,vocab_size])) 
	}
biases = { 
	'h1' : tf.Variable(tf.random_normal([h1])),
	'h2' : tf.Variable(tf.random_normal([h2])),
	'h3' : tf.Variable(tf.random_normal([h3])),
	'out' : tf.Variable(tf.random_normal([vocab_size]))
	}

layer_1 = tf.add(tf.matmul(x,weights['h1']), biases['h1'])
layer_1 = tf.nn.sigmoid(layer_1)

layer_2 = tf.add(tf.matmul(layer_1,weights['h2']), biases['h2'])
layer_2 = tf.nn.sigmoid(layer_2)

layer_3 = tf.add(tf.matmul(layer_2,weights['h3']), biases['h3'])
layer_3 = tf.nn.sigmoid(layer_3)



# layer_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden)
# layer_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden)
# rnn_cell = tf.contrib.rnn.MultiRNNCell([layer_1, layer_2])

# outputs, states = tf.contrib.rnn.static_rnn(rnn_cell,x, dtype=tf.float32)

pred = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['out']), biases['out']))


# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost = tf.reduce_mean(tf.losses.mean_squared_error(pred,y))
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	acc_total = 0
	loss_total = 0
	step = 0
	while step < training_iters:
		# print("steps done:", step)
		for i in range(len(data['oh'])-1):
			x_data = data['oh'][i]
			y_data = data['oh'][i+1]
			x_data = np.reshape(np.array(x_data), [1,-1])
			y_data = np.reshape(np.array(y_data), [1,-1])
			# print(x_data.shape)
			# print(y_data.shape)
			_, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, pred], feed_dict={x:x_data, y:y_data})
			loss_total += loss
			acc_total += acc
		if (step+1) % display_step == 0:
			print("Iter=" + str(step+1) + ",Average Loss = {:.6f}".format(loss_total/display_step) + ", Average Accuracy= {:.2f}%".format(acc_total/display_step))
			acc_total = 0
			loss_total = 0
		step += 1
	saver.save(sess, 'output/model_4')