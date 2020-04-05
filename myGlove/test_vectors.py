import parser
import numpy as np
from scipy.spatial import distance

def get_word_to_vec_dict():
    f = open('vectors.txt')
    data = f.readlines()

    word_to_vec = {}

    for line in data:
        elems = line.split(' ')
        word = elems[0]

        # remove new line from last element
        last_elem = elems[-1]
        split = last_elem.split('\n')
        elems[-1] = split[0]

        # get vector
        vector = []
        for val in elems[1:]:
            vector.append(float(val))
        word_to_vec[word] = vector

    f.close()
    return word_to_vec

def k_nearest_neighbors(word_to_vec, k):
    word_to_neighbors = {}
    
    for key1 in word_to_vec.keys():
        dists = []
        index_to_word = {}
        index = 0
        for key2 in word_to_vec.keys():
            index_to_word[index] = key2
            # don't consider word as its own neighbor 
            if key1 == key2:
                continue
            # get distance to neighbor
            dist = distance.euclidean(np.array(word_to_vec[key1]), np.array(word_to_vec[key2]))
            dists.append(dist)
            index += 1
        # sort distances in ascending order and get k closest
        sorted = np.argsort(dists)
        neighbor_indices = np.argwhere(sorted < k)
        
        for val in neighbor_indices:
            neighbor = index_to_word[val[0]]
            if key1 not in word_to_neighbors:
                word_to_neighbors[key1] = [neighbor]
            else:
                word_to_neighbors[key1].append(neighbor)

    return word_to_neighbors

            
def neighbors_to_file(word_to_neighbors):
    f = open('nearest_neighbors.txt', 'w')
    
    for key in word_to_neighbors.keys():
        f.write(key + ' ::::: ' + str(word_to_neighbors[key][0]))
        for neighbor in word_to_neighbors[key][1:]:
            f.write(', ' + neighbor)
        f.write('\n')
    f.close()
        


def main():
   word_to_vec = get_word_to_vec_dict()
   print('hit')
   word_to_neighbors = k_nearest_neighbors(word_to_vec, 5)
   print('hit')
   neighbors_to_file(word_to_neighbors)
   

main()