import parser
import numpy as np


# updates text8. also updates file holding all phrases if indicated
def write_to_text8(read_from_file, food_file, do_label_to_food):
    f = open(read_from_file)
    data = f.readlines()
    
    label_to_food = {}

    food_f = open(food_file, 'w')

    for line in data:
        stripped_line = line.rstrip()
        words = stripped_line.split('|') # can change plit

        label = words[0]
        phrase = words[1]
        #label, phrase = get_food_and_label(words)

        if label in label_to_food.keys():
            if len(label_to_food[label]) < 250:
                label_to_food[label].append(phrase)
                if (do_label_to_food):
                    food_f.write(phrase + '::::' + label + '\n')

        else:
            label_to_food[label] = [phrase]
            if (do_label_to_food):
                food_f.write(phrase + '::::' + label +'\n')


    food_f.close()
    f.close()

    out_file = open('text8', 'w')
    for key in label_to_food.keys():
        for val in label_to_food[key]:
            out_file.write(val + '\n')
    out_file.close()


# given sting, parses and returns the label and phrase separately
def get_food_and_label(words):
    last_word = words[-1]
    
    last_word = last_word.split('\t')
    label = last_word[1]
    
    phrase = ' '.join(words[i] for i in range(len(words)))
    return label, phrase


# returns dictionary giving the label for every food
def get_food_labels(read_from_file):
    f = open(read_from_file)
    data = f.readlines()
    food_to_label = {}

    for line in data:
        stripped_line = line.rstrip()
        words = stripped_line.split(' ')

        label, phrase = get_food_and_label(words)


        food_to_label[phrase] = label
    
    return food_to_label


def main():
    write_to_text8('words.txt', 'foods.txt', True)

main()