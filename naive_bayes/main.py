#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains the main program to read data, run classifiers
   and print results to stdout.
   You do not need to change this file. You can add debugging code or code to
   help produce your report, but this code should not be run by default in
   your final submission.
   Brown CS142, Spring 2020
"""

import numpy as np
import pandas as pd
from models import NaiveBayes

# def get_credit():
#     """
#     Gets and preprocesses German Credit data
#     """
#     data = pd.read_csv('./data/german_numerical-binsensitive.csv') # Reads file - may change

#     # MONTH categorizing
#     data['month'] = pd.cut(data['month'],3, labels=['month_1', 'month_2', 'month_3'], retbins=True)[0]
#     # month bins: [ 3.932     , 26.66666667, 49.33333333, 72.        ]
#     a = pd.get_dummies(data['month'])
#     data = pd.concat([data, a], axis = 1)
#     data = data.drop(['month'], axis=1)

#     # CREDIT categorizing
#     data['credit_amount'] = pd.cut(data['credit_amount'], 3, labels=['cred_amt_1', 'cred_amt_2', 'cred_amt_3'], retbins=True)[0]
#     # credit bins: [  231.826,  6308.   , 12366.   , 18424.   ]
#     a = pd.get_dummies(data['credit_amount'])
#     data = pd.concat([data, a], axis = 1)
#     data = data.drop(['credit_amount'], axis=1)

#     for header in ['investment_as_income_percentage', 'residence_since', 'number_of_credits']:
#         a = pd.get_dummies(data[header], prefix=header)
#         data = pd.concat([data, a], axis = 1)
#         data = data.drop([header], axis=1)

#     # change from 1-2 classes to 0-1 classes
#     data['people_liable_for'] = data['people_liable_for'] -1
#     data['credit'] = -1*(data['credit']) + 2 # original encoding 1: good, 2: bad. we switch to 1: good, 0: bad

#     # balance dataset
#     data = data.reindex(np.random.permutation(data.index)) # shuffle
#     pos = data.loc[data['credit'] == 1]
#     neg = data.loc[data['credit'] == 0][:350]
#     combined = pd.concat([pos, neg])

#     y = data.iloc[:, data.columns == 'credit'].to_numpy()
#     x = data.drop(['credit', 'sex', 'age', 'sex-age'], axis=1).to_numpy()

#     # split into train and validation
#     X_train, X_val, y_train, y_val = x[:350, :], x[351:526, :], y[:350, :].reshape([350,]), y[351:526, :].reshape([175,])


#     return X_train, X_val, y_train, y_val


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
    print (train.shape)
    print (test.shape)
    X_train = train[:12000, :]
    X_val = train[:12000, :]

    y_train = test[:12000].reshape([12000,])
    y_val = test[12000:14721].reshape([2721,])
    # X_train, X_val, y_train, y_val = train[:12000, :], train[12000:14721, :], test[:12000, :].reshape([12000,]), test[12000:14721, :].reshape([2721,])
    return X_train, X_val, y_train, y_val




def main():

    np.random.seed(0)
    vocab, phrase_to_label, phrase_to_words = get_vocab('words.txt')
    X_train, X_val, y_train, y_val = build_train_and_test(vocab, phrase_to_label, phrase_to_words)
    # X_train, X_val, y_train, y_val, = get_credit()

    model = NaiveBayes(2)

    model.train(X_train, y_train)

    print("------------------------------------------------------------")

    print("Train accuracy:")
    print(model.accuracy(X_train, y_train))

    print("------------------------------------------------------------")

    print("Test accuracy:")
    print(model.accuracy(X_val, y_val))



if __name__ == "__main__":
    main()
