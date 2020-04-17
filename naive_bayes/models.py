import numpy as np

class NaiveBayes(object):
    """ Bernoulli Naive Bayes model
    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes model with n_classes. """
        self.labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99, 999]
        self.n_classes = len(self.labels)
        self.att_dist = None
        self.priors_dist = None

    def train(self, X_train, y_train):
        """ Trains the model, using maximum likelihood estimation.
        @params:
            X_train: a n_examples x n_attributes numpy array
            y_train: a n_examples numpy array
        @return:
            a tuple consisting of:
                1) a 2D numpy array of the attribute distributions
                2) a 1D numpy array of the priors distribution
        """
        # TODO
        print("Sample data: ")
        print(X_train[:10])
        # find priors distribution
        class_occ = {}

        # counts number of occurrences for each class
        for label in y_train:
            if label not in class_occ:
                class_occ[label] = 1
            else:
                class_occ[label] += 1

        for occ in class_occ:
            print("Label: ", occ, " with count ",class_occ[occ], " out of ", y_train.shape)

        num_examples = X_train.shape[0]
        # add number of classes to number of examples for smoothing
        denominator = num_examples + self.n_classes

        priors_dist = np.zeros((self.n_classes))

        for c in range(self.n_classes):
            label = self.labels[c]
            priors_dist[c] = (class_occ[label] + 1) / denominator # add 1 to numerator for smoothing

        # find attribute distribution
        num_atts = X_train.shape[1]
        att_dist = np.zeros((num_atts, self.n_classes))

        # for each attribute/feature
        for i in range(num_atts):
            att_occ_per_class = {}

            # initialize dict for number of times attribute appeares in the classes
            for c in self.labels:
                att_occ_per_class[c] = 0

            # for all examples
            for j in range(num_examples):
                # if attribute i present, add to number of occurrences for class of example
                if X_train[j][i] == 1:
                    att_occ_per_class[y_train[j]] += 1

            for k in range(self.n_classes):
                label = self.labels[k]
                att_dist[i][k] = (att_occ_per_class[label] + 1) / (class_occ[label] + 2)

        self.att_dist = att_dist
        self.priors_dist = priors_dist

        return (att_dist, priors_dist)

    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """
        # TODO
        predictions = np.zeros((len(inputs)))
        # for each input to predict
        for i in range(len(inputs)):
            # make copy of attribute distribution (so as to not alter the original)
            att_dist = np.copy(self.att_dist)
            # go through attributes in each input
            for j in range(len(inputs[i])):
                # if encountering a missing feature in example,
                # subtract prob from 1 in that row in the attribute distribution
                if inputs[i][j] == 0:
                    att_dist[j] = 1 - att_dist[j]

            sums = []
            # find sum of logs down each column
            for col in range(self.n_classes):
                sum = 0
                for row in range(len(att_dist)):
                    sum += np.log(att_dist[row][col])
                sums.append(np.exp(sum))

            # multiply column sum with prior distribution and pick biggest
            max_val = -float('inf')
            max_index = 0
            for k in range(len(sums)):
                sums[k] *= self.priors_dist[k]
                if sums[k] > max_val:
                    max_val = sums[k]
                    max_index = k

            predictions[i] = max_index

        return predictions



    def accuracy(self, X_test, y_test):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            X_test: 2D numpy array of examples
            y_test: numpy array of labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """

        # TODO
        predictions = self.predict(X_test)
        print ('post predict')
        num_correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                num_correct += 1
        return num_correct / len(predictions)
