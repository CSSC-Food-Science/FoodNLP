from copy import deepcopy
import random

def write_vec_file(tups):
    ts = list(map(lambda t: deepcopy(t), tups))
    random.shuffle(ts)
    df = open('./textfiles/words.txt', 'w')
    for pair in ts:
        df.write(str(pair[0]) + "|" + pair[1] + '\n')
    df.close()

def write_clusters(expected_clusters, predicted_clusters):
    fn = './output/clusters_expected.txt'
    df = open(fn, 'w')
    for code in expected_clusters:
        df.write(">>>>>>>>>> CLUSTER: " + str(code) + " \n")
        for description in expected_clusters[code]:
            df.write(description + "\n")
    df.close()

    fn = './output/clusters_predicted.txt'
    df = open(fn, 'w')
    for code in predicted_clusters:
        df.write(">>>>>>>>>> CLUSTER: " + str(code) + " \n")
        for description in predicted_clusters[code]:
            df.write(description + "\n")
    df.close()

def write_all_cluster_analysis(analysis):
    fn = './output/clusters_analysis.txt'
    df = open(fn, 'w')
    for pred_key in analysis:
        df.write(">>>>> Prediction Cluster:" + str(pred_key) + "\n")
        for exp_key in analysis[pred_key]:
            (captured, miss) = analysis[pred_key][exp_key]
            df.write("Cluster analysis for exp:" + str(exp_key) + " and pred:" + str(pred_key) + "\n")
            df.write("exp. accuracy = " + str(captured) + "\n")
            df.write("pred. miss = " + str(miss) + "\n")
    df.close()

def write_selected_cluster_analysis(results_acc, results_miss):
    fn = './output/selected_clusters_analysis.txt'
    df = open(fn, 'w')
    for key in results_acc:
        cap = results_acc[key]
        miss = results_miss[key]
        df.write("Cluster analysis for: " + key + "\n")
        df.write("exp. accuracy = " + str(cap) + "\n")
        df.write("pred. miss = " + str(miss) + "\n")
    df.close()