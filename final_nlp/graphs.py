import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def create_acc_graphs(results_acc):
    X = []
    Y = []
    for key in results_acc:
        X.append(key)
        v = results_acc[key]
        Y.append(v)

    plt.ylim(0.0,1.0)
    plt.bar(X,Y)
    plt.title('expected_clusters/predicted_clusters accuracies')
    plt.suptitle('intersection of clusters over expected')
    plt.savefig('./images/accuracies.png')
    plt.clf()

def create_miss_graphs(results_miss):
    X = []
    Y = []
    for key in results_miss:
        X.append(key)
        v = results_miss[key]
        Y.append(v)
    plt.ylim(0.0,1.0)
    plt.bar(X,Y)
    plt.title('expected_clusters/predicted_clusters misses')
    plt.suptitle('predicted minus the intersection of clusters')
    plt.savefig('./images/misses.png')
    plt.clf()

def expected_clusters_distribution(expected_clusters):
    names = []
    labels = []
    
    for code in expected_clusters:
        names.append(str(code))
        size = len(expected_clusters[code])
        labels.append(size)

    plt.bar(names, labels)
    plt.title('Distribution of expected clusters')
    plt.savefig('./images/distribution_expected.png')
    plt.clf()

def predicted_clusters_distribution(predicted_clusters):
    names = []
    labels = []
    for code in predicted_clusters:
        names.append(str(code))
        size = len(predicted_clusters[code])
        labels.append(size)

    plt.bar(names,labels)
    plt.title('Distribution of predicted clusters')
    plt.savefig('./images/distribution_predicted.png')
    plt.clf()

def measure_inertia(X):
    y_line = []
    x_line = []
    for K in range(3,17):
        # TODO: possibly standardize the data before
        kmeans = KMeans(n_clusters=K, random_state=0)
        kmeans.fit(np.array(X))
        y_line.append(kmeans.inertia_)
        x_line.append(K)

    plt.plot(x_line, y_line)
    plt.ylabel('Inertia')
    plt.xlabel('Clusters')
    plt.title('Clusters vs inertia for dataset')
    plt.savefig('./images/inertia_measures.png')
    plt.clf()