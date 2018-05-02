import pandas  as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class NewsChecker():
    ALGORITHM_MultinomialNB = 0
    ALGORITHM_KNeighborsClassifier = 1

    @staticmethod
    def fake_or_real(news, algorithm):
        df = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv")

        label = df["label"]
        feature = df['text']

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(feature, label, test_size=0.1)

        X_data = pd.Series([news])

        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

        train = vectorizer.fit_transform(X_train)
        data = vectorizer.transform(X_data)

        if algorithm == NewsChecker.ALGORITHM_MultinomialNB:
            clf = MultinomialNB()
            clf.fit(train, y_train)
            predicted = clf.predict(data)

        else:
            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(train, y_train)
            predicted = neigh.predict(data)

        return predicted

#print(NewsChecker.fake_or_real('Tomorrow is Saturday', 0))
