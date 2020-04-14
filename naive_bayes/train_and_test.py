import numpy as np

def get_vocab(filename):
    f = open(filename) # use words.txt   
    data = f.readlines()

    vocab = []
    phrase_to_label = {}
    phrase_to_words = {}

    for line in data:
        split_line = line.rstrip().split('|')
        words = split_line[1].split(' ')
        
        phrase_to_label[split_line[1]] = split_line[0]
        phrase_to_words[split_line[1]] = words

        for word in words:
            if word not in vocab:
                vocab.append(word)


    f.close()

    return vocab, phrase_to_label, phrase_to_words

def build_train_and_test(vocab, phrase_to_label, phrase_to_words):
    num_words = len(vocab)
    num_examples = len(phrase_to_label.keys())

    train = np.zeros((num_examples, num_words))
    test = np.zeros(num_examples)

    i = 0
    for key in phrase_to_label.keys():
        for j in range(num_words):
            if vocab[j] in phrase_to_words[key]:
                train[i][j] = 1
            else:
                train[i][j] = 0
        test[i] = phrase_to_label[key]

        i += 1
    print (train)
    print (test)
    return train, test
        



def main():
    vocab, phrase_to_label, phrase_to_words = get_vocab('words.txt')
    train, test = build_train_and_test(vocab, phrase_to_label, phrase_to_words)

main()