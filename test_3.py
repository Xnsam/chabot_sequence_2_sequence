import numpy as np 
import collections
import random
import json
import time

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

np.random.seed(16)

data = {}
data['dump'] = []
with open('data.txt','r') as f:
	for line in f:
		data['dump'].append(line)

def make_data(data):
    content = [x.strip() for x in data]
    content = np.array(content)
    content = np.reshape(content, [-1,])
    return content

training_data = make_data(data['dump'])

def make_dictionary(words):
	count = collections.Counter(words).most_common()
	data['diction'] = {}
	for word, _ in count:
		data['diction'][word] = len(data['diction'])
	data['rev_diction'] = dict(zip(data['diction'].values(), data['diction'].keys()))


make_dictionary(training_data)

vocab_size = len(data['diction'])

learning_rate = 0.001
# epochs = 500000 + 500000 * 0.25
epochs = 10
# d_steps = 10000
n_input = 3
n_hidden = 512
model_path = 'output/'


symbols_in_keys = [[data['diction'][str(training_data[i])]] for i in range(len(training_data))]
symbols_in_keys = np.array(symbols_in_keys)
symbols_out_onehot = np.zeros([vocab_size], dtype=float)
for i in range(len(training_data)):
	symbols_out_onehot[data['diction'][str(training_data[i])]] = 1.0


model = Sequential()
model.add(LSTM(512, input_shape=(2,1), return_sequences=True))
model.add(LSTM(512, activation='sigmoid'))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

checkpoint = ModelCheckpoint(filepath=model_path+'model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

timestamp =  int(time.time())

with open( model_path + '%d-model.json' %timestamp, 'w') as f:
	d = json.loads(model.to_json())
	json.dump(d,f,indent=4)

symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1,2,1])
symbols_out_onehot = np.reshape(symbols_out_onehot, [1,-1])
model.fit(symbols_in_keys, symbols_out_onehot, batch_size=128, epochs=1, validation_split=0.2, callbacks=[checkpoint],shuffle=True,verbose=1)
model.save(model_path+'model.hdf5')
model.save_weights( model_path + '%d-weights-%f.hdf5' %(timestamp, scores[1]))

