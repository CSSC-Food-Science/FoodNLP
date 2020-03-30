from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from math import ceil, inf
import numpy as np
from copy import deepcopy
import random
from preprocess import preprocessGPQI
from write import write_vec_file, write_clusters, write_all_cluster_analysis,\
write_selected_cluster_analysis
from graphs import measure_inertia, predicted_clusters_distribution,\
expected_clusters_distribution, create_miss_graphs, create_acc_graphs

GPQI_fn = './data/masterGQPIcodes.csv'

def create_mappings(corpus, labels, codes):
    # label (code) -> descriptions
    # description -> label (code)
    codeDescriptions = {}
    descriptionLabel = {}

    codeDescriptionTuples = []

    # creating mappings
    for i in range(len(labels)):
        code = labels[i]
        description = corpus[i]
        codeDescriptionTuples.append((code, description))
        if code not in codeDescriptions:
            codeDescriptions[code] = []
        codeDescriptions[code].append(description)
        if description not in descriptionLabel:
            descriptionLabel[description] = code

    return codeDescriptions, descriptionLabel, codeDescriptionTuples

def get_vectors(fn, descrCode):
    df = open(fn, 'r')
    lines = df.readlines()

    pairs = []
    for line in lines:
        items = line.split("::::")
        description = items[0].lower()
        if description not in descrCode:
            print("ERROR: Couldn't find: " + description)
            continue
        code = descrCode[description]
        vec = items[1].replace(" ","").replace("[","").replace("]","").split(",")
        vector = list(map(lambda x: float(x), vec))
        pairs.append((description, vector, code))

    ps = list(map(lambda p: deepcopy(p), pairs))

    random.shuffle(ps)

    X = []
    Y = []
    Z = []

    for tuple in ps:
        code = tuple[2]
        vector = tuple[1]
        description = tuple[0]
        X.append(vector)
        Y.append(code)
        Z.append(description)

    return X, Y, Z, pairs

def get_predicted_clusters(kmeans, Z):
    predicted_clusters = {}
    for j in range(len(kmeans.labels_)):
        pred_label = kmeans.labels_[j]
        description = Z[j]
        if pred_label not in predicted_clusters:
            predicted_clusters[pred_label] = []
        predicted_clusters[pred_label].append(description)

    return predicted_clusters

def kmeans_and_glove(corpus, labels, codes):
    # lengths must be equal
    assert len(corpus) == len(labels)

    expected_clusters, descrCode, codDescrTup = create_mappings(corpus, labels, codes)
    write_vec_file(codDescrTup)

    vector_file = './data/final_results.txt' # WHAT IS THE FILE WITH THE VECTORS

    # Vectors
    X, Y, Z, P = get_vectors(vector_file, descrCode)

    # # # # #
    # TODO: possibly standardize the data before
    # # # # #
    X = StandardScaler().fit_transform(X)
    # # # # #

    measure_inertia(X)

    target_K = 13
    kmeans = KMeans(n_clusters=target_K, random_state=0)
    kmeans.fit(X)

    # cluster num -> predictions
    predicted_clusters = get_predicted_clusters(kmeans, Z)

    # write clusters
    write_clusters(expected_clusters, predicted_clusters)

    return expected_clusters, predicted_clusters

def analyze_clusters(expected_clusters, predicted_clusters):
    expected_clusters_distribution(expected_clusters)
    predicted_clusters_distribution(predicted_clusters)

    ids = {}
    descriptions = {}
    e_cluster = {}
    p_cluster = {}
    id = 1

    for cluster in predicted_clusters:
        for description in predicted_clusters[cluster]:
            if description not in ids:
                ids[description] = id
                descriptions[id] = description
                id += 1

    for cluster in expected_clusters:
        for description in expected_clusters[cluster]:
            if description not in ids:
                ids[description] = id
                descriptions[id] = description
                id += 1

    for cluster in predicted_clusters:
        items = predicted_clusters[cluster]
        p_cluster[cluster] = set(items) # TODO: convert to ids?

    for cluster in expected_clusters:
        items = expected_clusters[cluster]
        e_cluster[cluster] = set(items)

    analysis = {}
    for (pred_key, curr_predicted_cluster) in p_cluster.items():
        for (exp_key, curr_expected_cluster) in e_cluster.items():
            intersection = len(curr_expected_cluster & curr_predicted_cluster)
            captured = intersection / len(curr_expected_cluster)
            diff = len(curr_predicted_cluster) - intersection
            miss = diff / len(curr_predicted_cluster)
            if pred_key not in analysis:
                analysis[pred_key] = {}
            analysis[pred_key][exp_key] = (captured, miss)

    write_all_cluster_analysis(analysis)

    search = True
    results_acc = {}
    results_miss = {}
    selected_pairs = []
    while (search):
        if not analysis:
            search = False
            continue
        best_pk = None
        best_ek = None
        acc = -inf
        for pk in analysis:
            for ek in analysis[pk]:
                (cap, miss) = analysis[pk][ek]
                tval = (.5 * cap) + (.5 * miss)
                if tval > acc:
                    acc = cap
                    best_pk = pk
                    best_ek = ek

        (cap, miss) = analysis[best_pk][best_ek]
        selected_pairs.append((best_pk, best_ek))
        results_acc[str(best_ek) + "/" + str(best_pk)] = cap
        results_miss[str(best_ek) + "/" + str(best_pk)] = miss

        del analysis[best_pk]
        for p in analysis:
            del analysis[p][best_ek]

    predcodeExpcode = {}
    expcodePredcode = {}
    for pair in selected_pairs:
        pred = pair[0]
        exp = pair[1]
        predcodeExpcode[pred] = exp
        expcodePredcode[exp] = pred

    write_selected_cluster_analysis(results_acc, results_miss)
    create_acc_graphs(results_acc)
    create_miss_graphs(results_miss)
    return selected_pairs, predcodeExpcode, expcodePredcode

def predict(description, predcodeExpcode, expcodePredcode):
    # get the dictionary
    words = description.strip().split(" ")
    vector = None
    for word in words:
        if word in vectors:
            if vector == None:
                vector = vectors.get(word)
            else:
                vector += vector
        else:
            print("Not found: ", word)
    result = kmeans.predict(vector)
    # TODO: may need to convert to a numbers
    exp = predcodeExpcode[result]
    return exp

corpus, labels, codes = preprocessGPQI(GPQI_fn)
expected_clusters, predicted_clusters = kmeans_and_glove(corpus, labels, codes)
analyze_clusters(expected_clusters, predicted_clusters)
# Tries
# standardize
# remove labels