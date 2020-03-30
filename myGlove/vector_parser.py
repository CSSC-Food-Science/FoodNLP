import parser
import numpy as np



# returns all individual words in a phrase
def parse_food(read_from_file):
    f = open(read_from_file) # use words.txt   #open('foods.txt')
    data = f.readlines()

    phrases = {}

    for line in data:
        stripped_line = line.rstrip()
        words = stripped_line.split('|')
        phrases[words[1]] = words[1].split(' ') #stripped_line.split(' ')
        #phrases[stripped_line] = words

    f.close()

    return phrases

def parse_vectors(food_dict):
    f = open('vectors.txt')
    data = f.readlines()

    phrases_vectors = {}

    for key in food_dict.keys():
        phrases_vectors[key] = [0.0 for i in range(50)] 

    for line in data:
        stripped_line = line.rstrip()
        vals = stripped_line.split(' ')
        word = vals[0]
        vector = [float(i) for i in vals[1:]]

        # key is a phrase
        for key in food_dict.keys():
            # phrase_split = key.split(' ')
            # label, phrase = get_food_and_label(phrase_split)
            if word in food_dict[key]:
                phrases_vectors[key] = [int(phrases_vectors[key][i]) + vector[i] for i in range(len(vector))] 
    f.close()

    out_file = open('analysis/results/final_results.txt', 'w')

    for key in phrases_vectors.keys():
        # words = key.split(' ')
        # label, phrase = get_food_and_label(words)

        # label = food_to_label[phrase]
        out_file.write(key + ':::: [')
        # writes each component of vector
        for i in range(len(phrases_vectors[key])-1): 
            out_file.write(str(phrases_vectors[key][i]) + ', ')
        
        out_file.write(str(phrases_vectors[key][-1]) + '] \n')
    out_file.close()


def main():
    food_dict = parse_food('words.txt')
    parse_vectors(food_dict)

main()