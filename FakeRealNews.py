import pandas  as pd
import quandl
import math , datetime
import numpy as np
from sklearn import preprocessing ,cross_validation , svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import  pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from sklearn.datasets import make_blobs




df = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv")

label = df["label"]
feature = df['text']

X_train, X_test, y_train, y_test = cross_validation.train_test_split(feature, label, test_size=0.2 )

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

train = vectorizer.fit_transform(X_train)

test = vectorizer.transform(X_test)

print (test)

clf =  MultinomialNB()
clf.fit(train, y_train)
pred = clf.predict(test)

accuracy = metrics.accuracy_score(y_test, pred)

print("accuracy:  %0.3f" % accuracy)
print (pred)
