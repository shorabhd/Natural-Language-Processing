
# coding: utf-8

# In[3]:

import nltk
from gensim.models import word2vec
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import urllib.request
import sys


# In[4]:

def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'
    urllib.request.urlretrieve(url, 'test.txt')


# In[5]:

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


# In[6]:

def make_feature_dicts(data,w2v_model,token=True,caps=True,pos=True,
                       chunk=True,context=True,w2v=True):
    """
    Create feature dictionaries, one per token. Each entry in the dict consists of a key (a string)
    and a value of 1.
    Also returns a numpy array of NER tags (strings), one per token.
    See a3_test.
    The parameter flags determine which features to compute.
    Params:
    data.......the data returned by read_data
    token......If True, create a feature with key 'tok=X', where X is the *lower case* string for this token.
    caps.......If True, create a feature 'is_caps' that is 1 if this token begins with a capital letter.
               If the token does not begin with a capital letter, do not add the feature.
    pos........If True, add a feature 'pos=X', where X is the part of speech tag for this token.
    chunk......If True, add a feature 'chunk=X', where X is the chunk tag for this token
    context....If True, add features that combine all the features for the previous and subsequent token.
               E.g., if the prior token has features 'is_caps' and 'tok=a', then the features for the
               current token will be augmented with 'prev_is_caps' and 'prev_tok=a'.
               Similarly, if the subsequent token has features 'is_caps', then the features for the
               current token will also include 'next_is_caps'.
    Returns:
    - A list of dicts, one per token, containing the features for that token.
    - A numpy array, one per token, containing the NER tag for that token.
    """
    dicts = []
    labels = []
    for list in data:
        for t,tuple in enumerate(list):
            d = {}
            if(token):
                d['tok='+tuple[0].lower()] = 1
            if(caps):
                if tuple[0][0].isupper():
                    d['is_caps'] = 1
            if (pos):
                d['pos='+tuple[1]] = 1
            if (chunk):
                d['chunk='+tuple[2]] = 1
            if (context):
                if (t > 0):
                    if (token):
                        d['prev_tok='+list[t-1][0].lower()] = 1
                    if (caps):
                        if list[t - 1][0][0].isupper():
                            d['prev_is_caps'] = 1
                    if (pos):
                        d['prev_pos=' + list[t - 1][1]] = 1
                    if (chunk):
                        d['prev_chunk=' + list[t - 1][2]] = 1
                    if (w2v):
                        if list[t - 1][0] in w2v_model.wv.vocab.keys():
                            w2v_arr = w2v_model.wv[list[t - 1][0]]
                            for i in range(0,50):
                                d['prev_w2v_' + str(i)] = w2v_arr[i]
                if(t < len(list)-1):
                    if (token):
                        d['next_tok=' + list[t + 1][0].lower()] = 1
                    if (caps):
                        if list[t + 1][0][0].isupper():
                            d['next_is_caps'] = 1
                    if (pos):
                        d['next_pos=' + list[t + 1][1]] = 1
                    if (chunk):
                        d['next_chunk=' + list[t + 1][2]] = 1
                    if (w2v):
                        if list[t + 1][0] in w2v_model.wv.vocab.keys():
                            w2v_arr = w2v_model.wv[list[t + 1][0]]
                            for i in range(0,50):
                                d['next_w2v_' + str(i)] = w2v_arr[i]
            if (w2v):
                if tuple[0] in w2v_model.wv.vocab.keys():
                    w2v_arr = w2v_model.wv[tuple[0]]
                    for i in range(0,50):
                        d['w2v_' + str(i)] = w2v_arr[i]
            dicts.append(d)
            labels.append(tuple[3])
            #print()
    return dicts, np.asarray(labels)
    pass


# In[7]:

def confusion(true_labels, pred_labels):
    """
    Create a confusion matrix, where cell (i,j)
    is the number of tokens with true label i and predicted label j.
    Params:
      true_labels....numpy array of true NER labels, one per token
      pred_labels....numpy array of predicted NER labels, one per token
    Returns:
    A Pandas DataFrame (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
    See Log.txt for an example.
    """
    columns = set(true_labels)
    
    X = np.zeros((len(columns),len(columns)))
    for t,i in enumerate(sorted(columns)):
        for p,j in enumerate(sorted(columns)):
            for a, b in zip(true_labels, pred_labels):
                if a == i and b == j:
                    X[t][p] += 1
    X = X.astype(np.int32)
    return pd.DataFrame(X,columns=sorted(columns),index=sorted(columns))
    #return pd.DataFrame(pd.crosstab(true_labels,pred_labels),columns=sorted(columns),index=sorted(columns))
    pass


# In[8]:

def evaluate(confusion_matrix):
    """
    Compute precision, recall, f1 for each NER label.
    The table should be sorted in ascending order of label name.
    If the denominator needed for any computation is 0,
    use 0 as the result.  (E.g., replace NaNs with 0s).
    NOTE: you should implement this on your own, not using
          any external libraries (other than Pandas for creating
          the output.)
    Params:
      confusion_matrix...output of confusion function above.
    Returns:
      A Pandas DataFrame. See Log.txt for an example.
    """
    rows = ['precision','recall','f1']
    X = confusion_matrix.as_matrix()
    columns = confusion_matrix.keys()
    evaluation_matrix = np.zeros((len(rows), len(columns)))
    for i,row in enumerate(sorted(columns)):
        evaluation_matrix[0][i] = X[i][i] / np.sum(X[:,i])
        evaluation_matrix[1][i] = X[i][i] / np.sum(X[i,:])
        if (evaluation_matrix[0][i] + evaluation_matrix[1][i]) > 0:
            evaluation_matrix[2][i] = (2 * evaluation_matrix[0][i] * evaluation_matrix[1][i])                                   / (evaluation_matrix[0][i] + evaluation_matrix[1][i])
        else:
            evaluation_matrix[2][i] = 0
    return pd.DataFrame(evaluation_matrix,columns=sorted(columns),index=rows)
    pass


# In[9]:

def average_f1s(evaluation_matrix):
    """
    Returns:
    The average F1 score for all NER tags,
    EXCLUDING the O tag.
    """
    return np.average(evaluation_matrix.as_matrix()[2][:-1])
    pass


# In[10]:

def evaluate_combinations(train_data, test_data, w2v_model):
    """
    Run 16 different settings of the classifier,
    corresponding to the 16 different assignments to the
    parameters to make_feature_dicts:
    caps, pos, chunk, context
    That is, for one setting, we'll use
    token=True, caps=False, pos=False, chunk=False, context=False
    and for the next setting we'll use
    token=True, caps=False, pos=False, chunk=False, context=True
    For each setting, create the feature vectors for the training
    and testing set, fit a LogisticRegression classifier, and compute
    the average f1 (using the above functions).
    Returns:
    A Pandas DataFrame containing the F1 score for each setting,
    along with the total number of parameters in the resulting
    classifier. This should be sorted in descending order of F1.
    (See Log.txt).
    Note1: You may find itertools.product helpful for iterating over
    combinations.
    Note2: You may find it helpful to read the main method to see
    how to run the full analysis pipeline.
    """
    columns = ['f1','n_params', 'caps', 'pos', 'chunk', 'context','w2v']
    bool = [True, False]
    comb = product(bool, repeat=5)
    result = []
    for c in sorted(comb):
        temp = []
        dicts, labels = make_feature_dicts(train_data,w2v_model, caps=c[0], pos=c[1],
                                           chunk=c[2], context=c[3],w2v=c[4])
        vec = DictVectorizer()
        X = vec.fit_transform(dicts)
        clf = LogisticRegression()
        clf.fit(X, labels)
        test_dicts, test_labels = make_feature_dicts(test_data,w2v_model, caps=c[0], pos=c[1],
                                             chunk=c[2], context=c[3],w2v=c[4])
        X_test = vec.transform(test_dicts)
        preds = clf.predict(X_test)
        n_params = np.multiply(clf.coef_.shape[0],clf.coef_.shape[1])
        temp.append(average_f1s(evaluate(confusion(test_labels, preds))))
        temp.insert(1, n_params)
        temp.extend(c)
        result.append(temp)
    #return pd.DataFrame(sorted(result,key=lambda x:-x[0]), columns=columns, index=index)
    return pd.DataFrame(result, index=range(0, 32), columns=columns).sort_values(by='f1', axis=0, ascending=False)
    pass


# In[11]:

def main():
    """
    This method is done for you.
    See Log.txt for expected output.
    """
    
    download_data()
    brown = nltk.corpus.brown
    w2v_model = word2vec.Word2Vec(brown.sents(), size=50, window=5, min_count=5)
    w2v_model.save("w2v_model")
    w2v_model = word2vec.Word2Vec.load("w2v_model")
    train_data = read_data('train.txt')
    dicts, labels = make_feature_dicts(train_data,
                                       w2v_model,
                                       token=True,
                                       caps=True,
                                       pos=True,
                                       chunk=True,
                                       context=True,
                                       w2v=True)
    vec = DictVectorizer()
    X = vec.fit_transform(dicts)
    print('training data shape: %s\n' % str(X.shape))
    clf = LogisticRegression()
    clf.fit(X, labels)

    test_data = read_data('test.txt')
    test_dicts, test_labels = make_feature_dicts(test_data,
                                                 w2v_model,
                                                 token=True,
                                                 caps=True,
                                                 pos=True,
                                                 chunk=True,
                                                 context=True,
                                                 w2v=True)
    X_test = vec.transform(test_dicts)
    print('testing data shape: %s\n' % str(X_test.shape))

    preds = clf.predict(X_test)

    confusion_matrix = confusion(test_labels, preds)
    print('confusion matrix:\n%s\n' % str(confusion_matrix))

    evaluation_matrix = evaluate(confusion_matrix)
    print('evaluation matrix:\n%s\n' % str(evaluation_matrix))

    print('average f1s: %f\n' % average_f1s(evaluation_matrix))

    combo_results = evaluate_combinations(train_data, test_data, w2v_model)
    print('combination results:\n%s' % str(combo_results))
    
    f = open('output1.txt', 'w')
    f.write('combination results:\n%s' % str(combo_results))
    f.close()


# In[12]:

if __name__ == '__main__':
    #orig_stdout = sys.stdout
    #f = open('output1.txt', 'w')
    #sys.stdout = f
    main()
    #sys.stdout = orig_stdout
    #f.close()


# In[ ]:



