from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from preprocess import preprocessGPQI
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

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
    cv = CountVectorizer(stop_words='english',dtype=np.float32, ngram_range=(2, 2))
    X = cv.fit_transform(X)
    return X,cv

def count_vec(X):
    cv = CountVectorizer(stop_words='english',dtype=np.float32, binary=True)
    X = cv.fit_transform(X)
    return X,cv

def tfidf_vec(X):
    tf = TfidfVectorizer(stop_words='english',dtype=np.float32, ngram_range=(1, 2))
    X = tf.fit_transform(X)
    return X, tf

def bigram_tfidf_vec(X):
    tf = TfidfVectorizer(stop_words='english',dtype=np.float32, ngram_range=(2, 2))
    X = tf.fit_transform(X)
    return X, tf

def write_stuff(zips, filename):
    df = open('./textfiles/' + filename, 'w')
    for elem in all:
        input = elem[0]
        pred = elem[1][0]
        actual = elem[1][1]
        df.write(str(pair[0]) + "|" + pair[1] + '\n')
    df.close()

def graph_group(actual, pred, title, filename):
    print(title)
    pos = np.arange(11)
    positions = [1,2,3,4,5,6,7,8,9,10,11]#,99,999])
    width = 0.3

    bar1 = plt.bar(pos, actual, width, color="g")
    bar2 = plt.bar(pos + width, pred, width, color="r")
    plt.xticks(pos, positions)
    plt.title(title)
    plt.legend((bar1[0], bar2[0]), ('Actual', 'Predicted'))
    plt.savefig('./images/distributions_' + filename + '.png')
    plt.clf()

def test(title, model, func, filename):
    X, ff = func(filtered_corpus)
    x_train = X[:train]
    y_train = filtered_labels[:train]
    x_test = X[train:]
    y_test = filtered_labels[train:]

    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = model.score(x_test, y_test)

    zipped = list(zip(y_pred,y_test))
    all = list(zip(x_test, zipped))

    real_values = {}
    pred_values = {}

    # write_vec_file(all, filename)
    positions = [1,2,3,4,5,6,7,8,9,10,11]
    for position in positions:
        real_values[position] = 0
        pred_values[position] = 0

    for elem in all:
        input = elem[0]

        pred = elem[1][0]
        actual = elem[1][1]

        if actual not in real_values:
            real_values[actual] = 0
        real_values[actual] += 1

        if pred not in pred_values:
            pred_values[pred] = 0
        pred_values[pred] += 1

    actual = np.array(list(map(lambda x: x[1], sorted(real_values.items(), key=lambda item: item[0]))))
    print(actual)
    predicted = np.array(list(map(lambda x: x[1], sorted(pred_values.items(), key=lambda item: item[0]))))
    print(predicted)

    graph_group(actual, predicted, title, filename)
    print(title, acc)
    mae = metrics.mean_absolute_error(y_test, y_pred)

    mse = metrics.mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)
    return acc,mae,mse,rmse


def run():
    data= [
        ["Model","Vectorizer","Accuracy","MAE","MSE","RMSE"],
        ["LinearSVC","CountVectorizer", 0,0,0,0],
        ["LinearSVC","TFIDF", 0,0,0,0],
        ["LinearSVC","Bigram", 0,0,0,0],
        ["LinearSVC","Bigram/TFIDF", 0,0,0,0],
        ["SVC","CountVectorizer", 0,0,0,0],
        ["SVC","TFIDF", 0,0,0,0],
        ["SVC","Bigram", 0,0,0,0],
        ["SVC","Bigram/TFIDF", 0,0,0,0],
        ["Multinomial Naive Bayes","CountVectorizer", 0,0,0,0],
        ["Multinomial Naive Bayes","TFIDF", 0,0,0,0],
        ["Multinomial Naive Bayes","Bigram", 0,0,0,0],
        ["Multinomial Naive Bayes","Bigram/TFIDF", 0,0,0,0],
    ]
    lsvc = LinearSVC()
    acc,mae,mse,rmse = test('LinearSVC + CV: ', lsvc, count_vec, 'linear_svc_and_cv')
    data[1][2] = acc
    data[1][3] = mae
    data[1][4] = mse
    data[1][5] = rmse

    lscv2 =  LinearSVC()
    acc,mae,mse,rmse = test('LinearSVC + tfidf: ', lscv2, tfidf_vec, 'linear_svc_and_tfidf')
    data[2][2] = acc
    data[2][3] = mae
    data[2][4] = mse
    data[2][5] = rmse

    lscv3 =  LinearSVC()
    acc,mae,mse,rmse = test('LinearSVC + BG: ', lscv3, bigram_vec, 'linear_svc_and_bg')
    data[3][2] = acc
    data[3][3] = mae
    data[3][4] = mse
    data[3][5] = rmse

    lscv4 = LinearSVC()
    acc,mae,mse,rmse = test('LinearSVC + BGtfidf: ', lscv4, bigram_tfidf_vec, 'linear_svc_and_bgtfidf')
    data[4][2] = acc
    data[4][3] = mae
    data[4][4] = mse
    data[4][5] = rmse

    svc = SVC()
    acc,mae,mse,rmse = test('SVC + CV: ', svc, count_vec, 'svc_and_cv')
    data[5][2] = acc
    data[5][3] = mae
    data[5][4] = mse
    data[5][5] = rmse

    svc2 = SVC()
    acc,mae,mse,rmse = test('SVC + tfidf: ', svc2, tfidf_vec,'svc_and_tfidf')
    data[6][2] = acc
    data[6][3] = mae
    data[6][4] = mse
    data[6][5] = rmse

    svc3 = SVC()
    acc,mae,mse,rmse = test('SVC + BG: ', svc3, bigram_vec,'svc_and_bg')
    data[7][2] = acc
    data[7][3] = mae
    data[7][4] = mse
    data[7][5] = rmse

    svc4 = SVC()
    acc,mae,mse,rmse = test('SVC + BGtfidf: ', svc4, bigram_tfidf_vec, 'svc_and_bgtfidf')
    data[8][2] = acc
    data[8][3] = mae
    data[8][4] = mse
    data[8][5] = rmse

    mnnb1 = MultinomialNB()
    acc,mae,mse,rmse = test('MultNB + CV: ', mnnb1, count_vec, 'multnb_and_cv')
    data[9][2] = acc
    data[9][3] = mae
    data[9][4] = mse
    data[9][5] = rmse

    mnnb2 = MultinomialNB()
    acc,mae,mse,rmse = test('MultNB + tfidf: ', mnnb2, tfidf_vec, 'multnb_and_tfidf')
    data[10][2] = acc
    data[10][3] = mae
    data[10][4] = mse
    data[10][5] = rmse

    mnnb3 = MultinomialNB()
    acc,mae,mse,rmse = test('MultNB + BG: ', mnnb3, bigram_vec, 'multnb_and_bg')
    data[11][2] = acc
    data[11][3] = mae
    data[11][4] = mse
    data[11][5] = rmse

    mnnb4 = MultinomialNB()
    acc,mae,mse,rmse = test('MultNB + BGtfidf: ', mnnb4, bigram_tfidf_vec, 'multnb_and_bgtfidf')
    data[12][2] = acc
    data[12][3] = mae
    data[12][4] = mse
    data[12][5] = rmse

### TODO: Review these sus nums
    ### lr1 = LinearRegression()
    ### test("LinearReg + CV", lr1, count_vec)

    ### lr2 = LinearRegression()
    ###test("LinearReg + tfidf", lr2, tfidf_vec)

    ### lr2 = LinearRegression()
    ### test("LinearReg + BG", lr2, bigram_vec)

    ### lr2 = LinearRegression()
    ### test("LinearReg + BGtfidf", lr2, bigram_tfidf_vec)
    fig = plt.figure(figsize=(10,12), dpi=280)
    ax = fig.add_subplot(1,1,1)
    table = ax.table(cellText=data, loc='center')
    table.set_fontsize(16)
    table.scale(1,4)
    ax.axis('off')
    plt.savefig('./images/table_results.png')
run()