from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from preprocess import preprocessGPQI

GPQI_fn = './data/masterGQPIcodes.csv'

corpus, labels, codes = preprocessGPQI(GPQI_fn)

filtered_corpus = []
filtered_labels = []

for i in range(len(labels)):
    label = labels[i]
    text = corpus[i]
    if label != 999:
        filtered_corpus.append(text)
        filtered_labels.append(label)

sz = len(filtered_corpus) // 10
train = sz * 9

def count_vec(X):
    cv = CountVectorizer(stop_words='english',dtype = np.float32, ngram_range = (1, 2))
    X = cv.fit_transform(X)
    return X,cv

def tfidf_vec(X):
    tf = TfidfVectorizer()
    X = tf.fit_transform(X)
    return X, tf

def svm(x_train, y_train, x_test, y_test):
    clf = SVC()
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("Accuracy: ", score)

def test(title, model, func):
    X, ff = func(filtered_corpus)
    x_train = X[:train]
    y_train = filtered_labels[:train]
    x_test = X[train:]
    y_test = filtered_labels[train:]

    model = model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(title, acc)

def run():
    lsvc = LinearSVC()
    test('LinearSVC + CV: ', lsvc, count_vec)

    lscv2 =  LinearSVC()
    test('LinearSVC + tfidf: ', lscv2, tfidf_vec)

    svc = SVC()
    test('SVC + CV: ', svc, count_vec)

    svc2 = SVC()
    test('SVC + tfidf: ', svc, tfidf_vec)





run()