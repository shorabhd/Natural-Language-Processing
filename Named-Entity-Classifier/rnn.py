
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import tensorflow as tf
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
from gensim.models import word2vec
import sklearn as sk
from tensorflow.contrib import rnn
import nltk
import sys
from utils import *


# In[2]:

def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'
    urllib.request.urlretrieve(url, 'test.txt')


# In[3]:

def read_data(filename):
    """
    Read the data file into a list of lists of tuples.
    
    Each sentence is a list of tuples.
    Each tuple contains four entries:
    - the token
    - the part of speech
    - the phrase chunking tag
    - the named entity tag
    
    For example, the first two entries in the
    returned result for 'train.txt' are:
    
    > train_data = read_data('train.txt')
    > train_data[:2]
    [[('EU', 'NNP', 'I-NP', 'I-ORG'),
      ('rejects', 'VBZ', 'I-VP', 'O'),
      ('German', 'JJ', 'I-NP', 'I-MISC'),
      ('call', 'NN', 'I-NP', 'O'),
      ('to', 'TO', 'I-VP', 'O'),
      ('boycott', 'VB', 'I-VP', 'O'),
      ('British', 'JJ', 'I-NP', 'I-MISC'),
      ('lamb', 'NN', 'I-NP', 'O'),
      ('.', '.', 'O', 'O')],
     [('Peter', 'NNP', 'I-NP', 'I-PER'), ('Blackburn', 'NNP', 'I-NP', 'I-PER')]]
    """
    with open(filename) as f:
        lines = f.readlines()
    result = []
    temp = []
    for l,line in enumerate(lines):
        if line is not '\n':
            if '-DOCSTART-' not in line:
                temp.append(tuple([token.strip() for token in line.split()]))
        else:
            result.append(temp)
            temp = []
    result.append(temp)
    return result
    pass


# In[4]:

def make_w2v(data, tag_to_num):
    X = []
    Y = []
    brown = nltk.corpus.brown
    w2v_model = word2vec.Word2Vec(brown.sents(), size=300, window=5, min_count=5)
    for l in data:
        for t,tup in enumerate(l):
            y = np.zeros((len(tag_to_num), 1))
            if tup[0] in w2v_model.wv.vocab.keys():
                w2v_arr = w2v_model.wv[tup[0]]
                X.append(w2v_arr[:300])
                y[tag_to_num[tup[3]]] = 1
                Y.append(y)
    X = np.asarray(X)
    X = np.reshape(X,(X.shape[0],X.shape[1]))
    Y = np.asarray(Y)
    Y = np.reshape(Y,(Y.shape[0],Y.shape[1]))
    #print(X.shape, Y.shape)
    return X,Y


# In[5]:

def make_one_hot(data, tag_to_num):
    vocab = {}
    ner = {}
    X = []
    Y = []
    for l in data:
        for t,tup in enumerate(l):
            vocab[tup[0].lower()] = 1
            ner[tup[3]] = 1
    n_chars = len(vocab)  
    char2int = dict()
    int2char = dict()
    for i, c in enumerate(sorted(vocab.keys())):
        char2int[c] = i
        int2char[i] = c
    for l in data:
        x = np.zeros((n_chars, 1))
        y = np.zeros((len(tag_to_num), 1))
        for t,tup in enumerate(l):
            x[char2int[tup[0].lower()]] = 1
            X.append(x)
            y[tag_to_num[tup[3]]] = 1    
            Y.append(y)
    X = np.asarray(X)
    #print(X.shape)
    X = np.reshape(X,(X.shape[0],X.shape[1]))
    Y = np.asarray(Y)
    #print(Y.shape)
    Y = np.reshape(Y,(Y.shape[0],Y.shape[1]))
    return X, Y


# In[6]:

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # print(x.get_shape())
    x = tf.unstack(x, n_steps, 1)
    # print(np.array(x).shape)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # print(np.array(outputs).shape)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# In[7]:

def confusion(true_labels, pred_labels, columns):
	columns = sorted(columns)
	X = np.zeros((len(columns),len(columns)))
	for true,i in enumerate(sorted(columns)):
		for pred,j in enumerate(sorted(columns)):
			for a, b in zip(true_labels, pred_labels):
				if a == i and b == j:
					X[true][pred] += 1
	return pd.DataFrame(X.astype(np.int32), columns, columns)

def cal_confusion(pred_ids, y_ids, num_to_tag):
	columns = []
	for i in range(0,5):
		columns.append(num_to_tag[i])
	pred_labels = []
	true_labels = []
	for i in pred_ids:
		pred_labels.append(num_to_tag[i])
	for i in y_ids:
		true_labels.append(num_to_tag[i])
	return confusion(true_labels, pred_labels, columns)

def evaluate(confusion_matrix, num_to_tag):
	precision = []
	recall = []
	f1 = []
	columns = []
	for i in range(0,5):
		columns.append(num_to_tag[i])
	for idx, item in enumerate(sorted(columns)):
		p = np.diag(confusion_matrix)[idx] / np.sum(confusion_matrix, axis=0)[item]
		r = np.diag(confusion_matrix)[idx] / np.sum(confusion_matrix, axis = 1)[item]
		f = 0
		if p + r > 0:
			f = 2 * (p * r) / (p + r)
		f1.append(f)
		recall.append(r)
		precision.append(p)
	return pd.DataFrame(np.array([precision,recall,f1]),index=['precision', 'recall', 'f1'], columns=sorted(columns))

def average_f1s(evaluation_matrix, num_to_tag):
	"""
	Returns:
	The average F1 score for all NER tags,
	EXCLUDING the O tag.
	"""
	f1 = 0.0
	count = 0
	result = []
	for i in range(0,5):
		result.append(num_to_tag[i])
	for idx, item in enumerate(sorted(result)):
		if item.lower() != 'o':
			f1 += evaluation_matrix.iloc[2][idx]
			count += 1
	return f1 / count


# In[8]:

# Parameters
learning_rate = 10
training_iters = 10000
batch_size = 64
display_step = 10

# Network Parameters
n_input = 15
n_steps = 20
n_hidden = 20 
n_classes = 5 


# In[9]:

# tf Graph input
x = tf.placeholder("float32", [None, n_steps, n_input])
y = tf.placeholder("float32", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[10]:

pred = RNN(x, weights, biases)


# In[11]:

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# In[12]:

def main():
    #download_data()
    tagnames = ['I-LOC', 'I-MISC', 'I-ORG', 'I-PER','O']
    num_to_tag = dict(enumerate(tagnames))
    tag_to_num = {v:k for k,v in num_to_tag.items()}
    X,Y = make_w2v(read_data('train.txt'),tag_to_num)
    print("training data shape: ",X.shape)
    X_test, Y_test = make_w2v(read_data('test.txt'),tag_to_num)
    print("testing data shape: ",X_test.shape)
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x = X[step * batch_size : step * batch_size + batch_size] 
            batch_y = Y[step * batch_size : step * batch_size + batch_size] 
            #print(batch_x.shape)
            batch_x = batch_x.reshape((batch_size,n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                #print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1
        test_data = X_test.reshape((X_test.shape[0], n_steps, n_input))
        test_label = Y_test
        y_p = tf.argmax(pred, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_data, y:test_label})
        y_true = np.argmax(test_label,1)
        confusion_matrix = cal_confusion(y_pred, y_true, num_to_tag)
        print('confusion matrix:\n%s\n' % str(confusion_matrix))
        evaluation_matrix = evaluate(confusion_matrix, num_to_tag)
        print('evaluation matrix:\n%s\n' % str(evaluation_matrix))
        average_f1s_result = average_f1s(evaluation_matrix, num_to_tag)
        print('average f1s: %s\n' % str(average_f1s_result))


# In[13]:

if __name__=='__main__':
    orig_stdout = sys.stdout
    f = open('output2.txt', 'w')
    sys.stdout = f
    main()
    sys.stdout = orig_stdout
    f.close()


# In[ ]:



