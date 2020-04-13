import numpy as np

class NaiveBayes(object):
    """ Bernoulli Naive Bayes model
    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes model with n_classes. """
        self.n_classes = n_classes
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

        # find priors distribution
        class_occ = {}
        # counts number of occurrences for each class
        for label in y_train:
            if label not in class_occ:
                class_occ[label] = 1 
            else:
                class_occ[label] += 1

        num_examples = X_train.shape[0]
        # add number of classes to number of examples for smoothing
        denominator = num_examples + self.n_classes 

        priors_dist = np.zeros((self.n_classes))

        for c in range(self.n_classes):
           priors_dist[c] = (class_occ[c] + 1) / denominator # add 1 to numerator for smoothing
        
        # find attribute distribution
        num_atts = X_train.shape[1]
        att_dist = np.zeros((num_atts, self.n_classes))
        
        # for each attribute/feature
        for i in range(num_atts): 
            att_occ_per_class = {}
            
            # initialize dict for number of times attribute appeares in the classes
            for c in range(self.n_classes):
                att_occ_per_class[c] = 0

            # for all examples
            for j in range(num_examples):
                # if attribute i present, add to number of occurrences for class of example
                if X_train[j][i] == 1:
                    att_occ_per_class[y_train[j]] += 1

            for k in range(self.n_classes): 
                att_dist[i][k] = (att_occ_per_class[k] + 1) / (class_occ[k] + 2)

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

        num_correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                num_correct += 1
        return num_correct / len(predictions)

    def print_fairness(self, X_test, y_test, x_sens):
        """ 
        ***DO NOT CHANGE what we have implemented here.***
        
        Prints measures of the trained model's fairness on a given dataset (data).

        For all of these measures, x_sens == 1 corresponds to the "privileged"
        class, and x_sens == 1 corresponds to the "disadvantaged" class. Remember that
        y == 1 corresponds to "good" credit. 

        @params:
            X_test: 2D numpy array of examples
            y_test: numpy array of labels
            x_sens: numpy array of sensitive attribute values
        @return:

        """
        predictions = self.predict(X_test)

        # Disparate Impact (80% rule): A measure based on base rates: one of
        # two tests used in legal literature. All unprivileged classes are
        # grouped together as values of 0 and all privileged classes are given
        # the class 1. . Given data set D = (X,Y, C), with protected
        # attribute X (e.g., race, sex, religion, etc.), remaining attributes Y,
        # and binary class to be predicted C (e.g., “will hire”), we will say
        # that D has disparate impact if:
        # P[Y^ = 1 | S != 1] / P[Y^ = 1 | S = 1] <= (t = 0.8). 
        # Note that this 80% rule is based on US legal precedent; mathematically,
        # perfect "equality" would mean

        di = np.mean(predictions[np.where(x_sens==0)])/np.mean(predictions[np.where(x_sens==1)])
        print("Disparate impact: " + str(di))

        # Group-conditioned error rates! False positives/negatives conditioned on group
        
        pred_priv = predictions[np.where(x_sens==1)]
        pred_unpr = predictions[np.where(x_sens==0)]
        y_priv = y_test[np.where(x_sens==1)]
        y_unpr = y_test[np.where(x_sens==0)]

        # s-TPR (true positive rate) = P[Y^=1|Y=1,S=s]
        priv_tpr = np.sum(np.logical_and(pred_priv == 1, y_priv == 1))/np.sum(y_priv)
        unpr_tpr = np.sum(np.logical_and(pred_unpr == 1, y_unpr == 1))/np.sum(y_unpr)

        # s-TNR (true negative rate) = P[Y^=0|Y=0,S=s]
        priv_tnr = np.sum(np.logical_and(pred_priv == 0, y_priv == 0))/(len(y_priv) - np.sum(y_priv))
        unpr_tnr = np.sum(np.logical_and(pred_unpr == 0, y_unpr == 0))/(len(y_unpr) - np.sum(y_unpr))

        # s-FPR (false positive rate) = P[Y^=1|Y=0,S=s]
        priv_fpr = 1 - priv_tnr 
        unpr_fpr = 1 - unpr_tnr 

        # s-FNR (false negative rate) = P[Y^=0|Y=1,S=s]
        priv_fnr = 1 - priv_tpr 
        unpr_fnr = 1 - unpr_tpr

        print("FPR (priv, unpriv): " + str(priv_fpr) + ", " + str(unpr_fpr))
        print("FNR (priv, unpriv): " + str(priv_fnr) + ", " + str(unpr_fnr))
    
    
        # #### ADDITIONAL MEASURES IF YOU'RE CURIOUS #####

        # Calders and Verwer (CV) : Similar comparison as disparate impact, but
        # considers difference instead of ratio. Historically, this measure is
        # used in the UK to evalutate for gender discrimination. Uses a similar
        # binary grouping strategy. Requiring CV = 1 is also called demographic
        # parity.

        cv = 1 - (np.mean(predictions[np.where(x_sens==1)]) - np.mean(predictions[np.where(x_sens==0)]))

        # Group Conditioned Accuracy: s-Accuracy = P[Y^=y|Y=y,S=s]

        priv_accuracy = np.mean(predictions[np.where(x_sens==1)] == y_test[np.where(x_sens==1)])
        unpriv_accuracy = np.mean(predictions[np.where(x_sens==0)] == y_test[np.where(x_sens==0)])

        return predictions
