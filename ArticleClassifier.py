from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

# -----------------------------------------------
# #NB Classifier
# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
# text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()) ])
# text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
# predicted = text_clf.predict(twenty_test.data)
# np.mean(predicted == twenty_test.target)
#
# print "\n","mean","\n"
# print (np.mean(predicted == twenty_test.target))

#--------------------------------------------------------
#Support Vector Machines (SVM)

text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
print np.mean(predicted_svm == twenty_test.target)