from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
from preprocess import preprocessGPQI

GPQI_fn = './data/masterGQPIcodes.csv'

corpus, labels, codes = preprocessGPQI(GPQI_fn)

filtered_corpus = []
filtered_labels = []
labels_set = set()
CLUSTERS = 11

for i in range(len(labels)):
    label = labels[i]
    text = corpus[i]

    if label != 999 and label != 99:
        filtered_corpus.append(text)
        filtered_labels.append(label)
        labels_set.add(label)

sz = len(filtered_corpus) // 10
train = sz * 9

def bigram_vec(X):
    # cv = CountVectorizer(stop_words='english',dtype=np.float32, ngram_range=(1, 2))
    # -> LinearSVC + BG:  0.4928909952606635
    # -> SVC + BG:  0.44008124576844954

    cv = CountVectorizer(stop_words='english',dtype=np.float32, ngram_range=(2, 2))
    X = cv.fit_transform(X)
    return X,cv

def count_vec(X):
    # cv = CountVectorizer(stop_words='english',dtype=np.float32, binary=True)
    # -> LinearSVC + CV:  0.4719025050778605
    # -> SVC + CV:  0.46445497630331756

    cv = CountVectorizer(stop_words='english',dtype=np.float32, binary=True)
    X = cv.fit_transform(X)
    return X,cv

def tfidf_vec(X):
    # tf = TfidfVectorizer()
    # -> SVC + tfidf:  0.4651320243737305
    # -> LinearSVC + tfidf:  0.48408937034529453

    tf = TfidfVectorizer(stop_words='english',dtype=np.float32, ngram_range=(1, 2))
    X = tf.fit_transform(X)
    return X, tf

def bigram_tfidf_vec(X):
    tf = TfidfVectorizer(stop_words='english',dtype=np.float32, ngram_range=(2, 2))
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
    # LinearSVC + CV:  0.5063913470993117

    lscv2 =  LinearSVC()
    test('LinearSVC + tfidf: ', lscv2, tfidf_vec)
    # LinearSVC + tfidf:  0.52015732546706

    lscv3 =  LinearSVC()
    test('LinearSVC + BG: ', lscv3, bigram_vec)
    # LinearSVC + BG:  0.30973451327433627

    lscv4 = LinearSVC()
    test('LinearSVC + BGtfidf: ', lscv4, bigram_tfidf_vec)
    # LinearSVC + BGtfidf:  0.30973451327433627


    svc = SVC()
    test('SVC + CV: ', svc, count_vec)
    # SVC + CV:  0.4631268436578171

    svc2 = SVC()
    test('SVC + tfidf: ', svc2, tfidf_vec)
    # SVC + tfidf:  0.4257620452310718

    svc3 = SVC()
    test('SVC + BG: ', svc3, bigram_vec)
    # SVC + BG:  0.24188790560471976

    svc4 = SVC()
    test('SVC + BGtfidf: ', svc4, bigram_tfidf_vec)
    # SVC + BGtfidf:  0.23402163225172073


    mnnb1 = MultinomialNB()
    test('MultNB + CV: ', mnnb1, count_vec)
    # MultNB + CV:  0.52015732546706
    mnnb2 = MultinomialNB()
    test('MultNB + tfidf: ', mnnb2, tfidf_vec)
    # MultNB + tfidf:  0.45624385447394294

    mnnb3 = MultinomialNB()
    test('MultNB + BG: ', mnnb3, bigram_vec)
    # MultNB + BG:  0.2949852507374631

    mnnb4 = MultinomialNB()
    test('MultNB + BGtfidf: ', mnnb4, bigram_tfidf_vec)
    # MultNB + BGtfidf:  0.2527040314650934

run()