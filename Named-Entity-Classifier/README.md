In this assignment, we will build a named-entity classifier
using LogisticRegression.

We'll use the labeled data from the CoNLL 2003 Shared Task:
http://www.cnts.ua.ac.be/conll2003/ner/

This is downloaded by the download_data method below.

The main goals of this assignment are to have you:
1- Implement different feature sets for the classifier.
2- Compute evaluation metrics for the classifier on the test set.
3- Enumerate over various settings of the features to determine
   which features result in the highest accuracy.

We will then train a word2vec model, then use it to (hopefully) improve the named-entity classifier. 