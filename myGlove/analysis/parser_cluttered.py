import parser
import numpy as np


def vectorize_for_jennifer():
    f = open('words.txt')
    data = f.readlines()

    foods = []

    for line in data:
        stripped_line = line.rstrip()
        words = stripped_line.split('|')
        foods.append(words[1])


    
    out_file = open('text8', 'w')
    for food in foods:
        out_file.write(food + '\n')
    
    out_file.close()
    



def update_text8():
    f = open('food_with_labels.txt')
    data = f.readlines()

    label_to_food = {}
    food_file = open('foods.txt', 'w')

    for line in data:
        stripped_line = line.rstrip()
        words = stripped_line.split(' ')
        # last_word = words[-1]
        
        # last_word = last_word.split('\t')

        # # words.append(last_word[0])
        # label = last_word[1]
        
        # phrase = ' '.join(words[i] for i in range(len(words)))
        label, phrase = get_food_and_label(words)


        if label in label_to_food.keys():
            if len(label_to_food[label]) < 250:
                label_to_food[label].append(phrase)
                food_file.write(phrase + '\n')

        else:
            label_to_food[label] = [phrase]
            food_file.write(phrase + '\n')

    food_file.close()
    f.close()

    out_file = open('text8', 'w')
    for key in label_to_food.keys():
        for val in label_to_food[key]:
            out_file.write(val + '\n')
    out_file.close()





def get_food_and_label(words):
    last_word = words[-1]
    
    last_word = last_word.split('\t')
    label = last_word[1]
    
    phrase = ' '.join(words[i] for i in range(len(words)))
    return label, phrase


def get_food_labels():
    f = open('food_with_labels.txt')
    data = f.readlines()
    food_to_label = {}

    for line in data:
        stripped_line = line.rstrip()
        words = stripped_line.split(' ')

        label, phrase = get_food_and_label(words)


        food_to_label[phrase] = label
    
    return food_to_label




def parse_food():
    f = open('words.txt') #open('foods.txt')
    data = f.readlines()

    phrases = {}

    for line in data:
        stripped_line = line.rstrip()
        words = stripped_line.split('|')
        phrases[words[1]] = words[1].split(' ') #stripped_line.split(' ')
        #phrases[stripped_line] = words

    f.close()

    return phrases

def new_parse_vectors(food_dict):
    f = open('vectors.txt')
    data = f.readlines()

    phrases_vectors = {}

    for key in food_dict.keys():
        phrases_vectors[key] = [0.0 for i in range(50)] #np.zeros(50)

    # print (phrases_vectors)
    for line in data:
        stripped_line = line.rstrip()
        vals = stripped_line.split(' ')
        word = vals[0]
        vector = [float(i) for i in vals[1:]]

        for key in food_dict.keys():
            # phrase_split = key.split(' ')
            # label, phrase = get_food_and_label(phrase_split)
            if word in food_dict[key]:
                phrases_vectors[key] = [int(phrases_vectors[key][i]) + vector[i] for i in range(len(vector))] #vector    # make sure doing element wise!
    f.close()

    out_file = open('results/final_results.txt', 'w')

    for key in phrases_vectors.keys():
        words = key.split(' ')
        # label, phrase = get_food_and_label(words)

        # label = food_to_label[phrase]
        out_file.write(key + ':::: [')
        for i in range(len(phrases_vectors[key])-1): 
            out_file.write(str(phrases_vectors[key][i]) + ', ')
        out_file.write(str(phrases_vectors[key][-1]) + '] \n')
    out_file.close()






def parse_vectors(food_dict, food_to_label):
    f = open('vectors.txt')
    data = f.readlines()

    phrases_vectors = {}

    for key in food_dict.keys():
        phrases_vectors[key] = [0.0 for i in range(50)] #np.zeros(50)

    # print (phrases_vectors)
    for line in data:
        stripped_line = line.rstrip()
        vals = stripped_line.split(' ')
        word = vals[0]
        vector = [float(i) for i in vals[1:]]
        print (vector)

        for key in food_dict.keys():
            phrase_split = key.split(' ')
            label, phrase = get_food_and_label(phrase_split)
            if word in food_dict[key]:
                phrases_vectors[key] = [int(phrases_vectors[key][i]) + vector[i] for i in range(len(vector))] #vector    # make sure doing element wise!
    f.close()

    out_file = open('results/final_results.txt', 'w')

    for key in phrases_vectors.keys():
        words = key.split(' ')
        label, phrase = get_food_and_label(words)

        # label = food_to_label[phrase]
        out_file.write(label + '::::' + phrase + ':::: [')
        for i in range(len(phrases_vectors[key])-1): 
            out_file.write(str(phrases_vectors[key][i]) + ', ')
        out_file.write(str(phrases_vectors[key][-1]) + '] \n')
    out_file.close()

    # psrint (phrases_vectors)



def main():
    # get_food_labels()
    # # update_text8()
    
    # food_to_label = get_food_labels()
    # parse_vectors(food_dict, food_to_label)

    # vectorize_for_jennifer()
    food_dict = parse_food()
    new_parse_vectors(food_dict)

main()
