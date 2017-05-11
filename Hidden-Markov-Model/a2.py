# coding: utf-8
<<<<<<< HEAD

# In[308]:


=======
>>>>>>> template/master
"""CS585: Assignment 2

In this assignment, you will complete an implementation of
a Hidden Markov Model and use it to fit a part-of-speech tagger.
"""

from collections import Counter, defaultdict
import math
import numpy as np
import os.path
import urllib.request


class HMM:
<<<<<<< HEAD
    def __init__(self, smoothing=0):
        """
        Construct an HMM model with add-k smoothing.
        Params:
          smoothing...the add-k smoothing value

        This is DONE.
        """
        self.smoothing = smoothing

    def fit_transition_probas(self, tags):
        """
        Estimate the HMM state transition probabilities from the provided data.

        Creates a new instance variable called `transition_probas` that is a
        dict from a string ('state') to a dict from string to float. E.g.
        {'N': {'N': .1, 'V': .7, 'D': 2},
         'V': {'N': .3, 'V': .5, 'D': 2},
         ...
        }
        See test_hmm_fit_transition.

        Params:
          tags...a list of lists of strings representing the tags for one sentence.
        Returns:
            None
        """
        states = sorted(set(np.concatenate(tags)))
        d = defaultdict(lambda: 0)
        for tag in tags:
            for t in range(len(tag) - 1):
                d[(tag[t], tag[t + 1])] += 1

        denom = defaultdict(lambda: 0)
        for key in d.keys():
            k1, k2 = key
            denom[(k1)] += d[key]

        transition_probas = defaultdict(lambda: 0)

        for a in states:
            transition_probas[a] = defaultdict(lambda: 0)
            temp = defaultdict(lambda: self.smoothing / (denom[a] + (self.smoothing * len(denom))))
            for b in states:
                temp[b] = (d[(a, b)] + self.smoothing) / (denom[a] + (self.smoothing * len(denom)))
            transition_probas[a] = temp;

        self.transition_probas = transition_probas
        self.states = states

    def fit_emission_probas(self, sentences, tags):
        """
        Estimate the HMM emission probabilities from the provided data.

        Creates a new instance variable called `emission_probas` that is a
        dict from a string ('state') to a dict from string to float. E.g.
        {'N': {'dog': .1, 'cat': .7, 'mouse': 2},
         'V': {'run': .3, 'go': .5, 'jump': 2},
         ...
        }

        Params:
          sentences...a list of lists of strings, representing the tokens in each sentence.
          tags........a list of lists of strings, representing the tags for one sentence.
        Returns:
            None

        See test_hmm_fit_emission.
        """
        d = defaultdict(lambda: 0)
        for sent, tag in zip(sentences, tags):
            for s, t in zip(sent, tag):
                d[(t, s)] += 1

        denom = defaultdict(lambda: 0)
        for key in d.keys():
            k1, k2 = key
            denom[(k1)] += d[key]

        emission_probas = defaultdict(lambda: 0)

        for k in sorted(denom.keys()):
            emission_probas[k] = defaultdict(lambda: 0)
            temp = defaultdict(lambda: self.smoothing / (denom[k] + (self.smoothing * len(d))))
            for key in d.keys():
                if key[0] == k:
                    temp[key[1]] = (d[key] + self.smoothing) / (denom[key[0]] + (self.smoothing * len(d)))
            emission_probas[k] = temp
        self.emission_probas = emission_probas

    def fit_start_probas(self, tags):
        """
        Estimate the HMM start probabilities form the provided data.

        Creates a new instance variable called `start_probas` that is a
        dict from string (state) to float indicating the probability of that
        state starting a sentence. E.g.:
        {
            'N': .4,
            'D': .5,
            'V': .1
        }

        Params:
          tags...a list of lists of strings representing the tags for one sentence.
        Returns:
            None

        See test_hmm_fit_start
        """
        d = defaultdict(lambda: 0)
        for tag in tags:
            d[tag[0]] += 1
        start_probas = defaultdict(lambda: 0)
        for s in self.states:
            start_probas[s] = (d[s] + self.smoothing) / (len(tags) + (self.smoothing * len(self.states)))
        self.start_probas = start_probas

    def fit(self, sentences, tags):
        """
        Fit the parameters of this HMM from the provided data.

        Params:
          sentences...a list of lists of strings, representing the tokens in each sentence.
          tags........a list of lists of strings, representing the tags for one sentence.
        Returns:
            None

        DONE. This just calls the three fit_ methods above.
        """
        self.fit_transition_probas(tags)
        self.fit_emission_probas(sentences, tags)
        self.fit_start_probas(tags)

    def viterbi(self, sentence):
        """
        Perform Viterbi search to identify the most probable set of hidden states for
        the provided input sentence.

        Params:
          sentence...a lists of strings, representing the tokens in a single sentence.

        Returns:
          path....a list of strings indicating the most probable path of POS tags for
                  this sentence.
          proba...a float indicating the probability of this path.
        """
        states = sorted(self.states)
        start_probas = self.start_probas
        transition_probas = self.transition_probas
        emission_probas = self.emission_probas
        path = []
        start = {}
        viterbi = []
        for s in states:
            start[s] = start_probas[s] * emission_probas[s][sentence[0]]
        state = max(start.keys(), key=lambda tag: start[tag])
        path.append(state)
        viterbi.append(start)
        for j in range(1, len(sentence)):
            prev = viterbi[-1]
            current = {}
            for s in states:
                best_prev = max(prev.keys(), key=lambda prevtag: \
                                    prev[prevtag] * transition_probas[prevtag][s] * emission_probas[s][sentence[j]])
                current[s] = prev[best_prev] * transition_probas[best_prev][s] * emission_probas[s][sentence[j]]
            state = max(current.keys(), key=lambda tag: current[tag])
            path.append(state)
            viterbi.append(current)
        prev = viterbi[-1]
        best_prev = max(prev.keys(), key=lambda prevtag: prev[prevtag])
        proba = prev[best_prev]
        return (path, proba)

def read_labeled_data(filename):
    """
    Read in the training data, consisting of sentences and their POS tags.

    Each line has the format:
    <token> <tag>

    New sentences are indicated by a newline. E.g. two sentences may look like this:
    <token1> <tag1>
    <token2> <tag2>

    <token1> <tag1>
    <token2> <tag2>
    ...

    See data.txt for example data.

    Params:
      filename...a string storing the path to the labeled data file.
    Returns:
      sentences...a list of lists of strings, representing the tokens in each sentence.
      tags........a lists of lists of strings, representing the POS tags for each sentence.
    """
    sentences = []
    tags = []
    with open(filename) as f:
        lines = f.readlines()
    sent = []
    tag = []
    for line in lines:
        if line != "\n":
            token = line.split(" ")
            sent.append(token[0])
            tag.append(token[1].strip())
        else:
            sentences.append(sent)
            tags.append(tag)
            sent = []
            tag = []
    return sentences, tags
    pass
=======
	def __init__(self, smoothing=0):
		"""
		Construct an HMM model with add-k smoothing.
		Params:
		  smoothing...the add-k smoothing value
		
		This is DONE.
		"""
		self.smoothing = smoothing

	def fit_transition_probas(self, tags):
		"""
		Estimate the HMM state transition probabilities from the provided data.

		Creates a new instance variable called `transition_probas` that is a 
		dict from a string ('state') to a dict from string to float. E.g.
		{'N': {'N': .1, 'V': .7, 'D': 2},
		 'V': {'N': .3, 'V': .5, 'D': 2},
		 ...
		}
		See test_hmm_fit_transition.
		
		Params:
		  tags...a list of lists of strings representing the tags for one sentence.
		Returns:
			None
		"""
		###TODO
		pass

	def fit_emission_probas(self, sentences, tags):
		"""
		Estimate the HMM emission probabilities from the provided data. 

		Creates a new instance variable called `emission_probas` that is a 
		dict from a string ('state') to a dict from string to float. E.g.
		{'N': {'dog': .1, 'cat': .7, 'mouse': 2},
		 'V': {'run': .3, 'go': .5, 'jump': 2},
		 ...
		}

		Params:
		  sentences...a list of lists of strings, representing the tokens in each sentence.
		  tags........a list of lists of strings, representing the tags for one sentence.
		Returns:
			None		  

		See test_hmm_fit_emission.
		"""
		###TODO
		pass

	def fit_start_probas(self, tags):
		"""
		Estimate the HMM start probabilities form the provided data.

		Creates a new instance variable called `start_probas` that is a 
		dict from string (state) to float indicating the probability of that
		state starting a sentence. E.g.:
		{
			'N': .4,
			'D': .5,
			'V': .1		
		}

		Params:
		  tags...a list of lists of strings representing the tags for one sentence.
		Returns:
			None

		See test_hmm_fit_start
		"""
		###TODO
		pass

	def fit(self, sentences, tags):
		"""
		Fit the parameters of this HMM from the provided data.

		Params:
		  sentences...a list of lists of strings, representing the tokens in each sentence.
		  tags........a list of lists of strings, representing the tags for one sentence.
		Returns:
			None		  

		DONE. This just calls the three fit_ methods above.
		"""
		self.fit_transition_probas(tags)
		self.fit_emission_probas(sentences, tags)
		self.fit_start_probas(tags)


	def viterbi(self, sentence):
		"""
		Perform Viterbi search to identify the most probable set of hidden states for
		the provided input sentence.

		Params:
		  sentence...a lists of strings, representing the tokens in a single sentence.

		Returns:
		  path....a list of strings indicating the most probable path of POS tags for
		  		  this sentence.
		  proba...a float indicating the probability of this path.
		"""
		###TODO
		pass


def read_labeled_data(filename):
	"""
	Read in the training data, consisting of sentences and their POS tags.

	Each line has the format:
	<token> <tag>

	New sentences are indicated by a newline. E.g. two sentences may look like this:
	<token1> <tag1>
	<token2> <tag2>

	<token1> <tag1>
	<token2> <tag2>
	...

	See data.txt for example data.

	Params:
	  filename...a string storing the path to the labeled data file.
	Returns:
	  sentences...a list of lists of strings, representing the tokens in each sentence.
	  tags........a lists of lists of strings, representing the POS tags for each sentence.
	"""
	###TODO
	pass
>>>>>>> template/master

def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/ty7cclxiob3ajog/data.txt?dl=1'
    urllib.request.urlretrieve(url, 'data.txt')

if __name__ == '__main__':
<<<<<<< HEAD
    """
    Read the labeled data, fit an HMM, and predict the POS tags for the sentence
    'Look at what happened'

    DONE - please do not modify this method.

    The expected output is below. (Note that the probability may differ slightly due
    to different computing environments.)

    $ python3 a2.py
    model has 34 states
        ['$', "''", ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '``']
    predicted parts of speech for the sentence ['Look', 'at', 'what', 'happened']
    (['VB', 'IN', 'WP', 'VBD'], 2.751820088075314e-10)
    """
    fname = 'data.txt'
    if not os.path.isfile(fname):
        download_data()
    sentences, tags = read_labeled_data(fname)

    model = HMM(.001)
    model.fit(sentences, tags)
    print('model has %d states' % len(model.states))
    print(model.states)
    sentence = ['Look', 'at', 'what', 'happened']
    print('predicted parts of speech for the sentence %s' % str(sentence))
    print(model.viterbi(sentence))
=======
	"""
	Read the labeled data, fit an HMM, and predict the POS tags for the sentence
	'Look at what happened'

	DONE - please do not modify this method.

	The expected output is below. (Note that the probability may differ slightly due
	to different computing environments.)

	$ python3 a2.py  
	model has 34 states
        ['$', "''", ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '``']
	predicted parts of speech for the sentence ['Look', 'at', 'what', 'happened']
	(['VB', 'IN', 'WP', 'VBD'], 2.751820088075314e-10)
	"""
	fname = 'data.txt'
	if not os.path.isfile(fname):
		download_data()
	sentences, tags = read_labeled_data(fname)

	model = HMM(.001)
	model.fit(sentences, tags)
	print('model has %d states' % len(model.states))
	print(model.states)
	sentence = ['Look', 'at', 'what', 'happened']
	print('predicted parts of speech for the sentence %s' % str(sentence))
	print(model.viterbi(sentence))
>>>>>>> template/master
