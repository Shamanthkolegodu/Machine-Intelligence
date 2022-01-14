from functools import update_wrapper
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_stumps=20):

        self.n_stumps = n_stumps
        self.stumps = []

    def fit(self, X, y):
        self.alphas = []

        sample_weights = np.ones_like(y) / len(y)
        for _ in range(self.n_stumps):

            st = DecisionTreeClassifier(
                criterion='entropy', max_depth=1, max_leaf_nodes=2)
            st.fit(X, y, sample_weights)
            y_pred = st.predict(X)

            self.stumps.append(st)

            error = self.stump_error(y, y_pred, sample_weights=sample_weights)
            alpha = self.compute_alpha(error)
            self.alphas.append(alpha)
            sample_weights = self.update_weights(
                y, y_pred, sample_weights, alpha)

        return self

    def stump_error(self, y, y_pred, sample_weights):
        n=len(y_pred)
        error=0
        for x in range(n):
            if y_pred[x]!=y[x]:
                error+=sample_weights[x]
        sum_of_sample_weights=np.sum(sample_weights)
        # print(error/sum_of_sample_weights)
        return error/sum_of_sample_weights
        pass

    def compute_alpha(self, error):
        eps = 1e-9
        alpha=(0.5)*(math.log((1-(error+eps))/(error+eps)))
        # print(alpha)
        return alpha
        pass

    def update_weights(self, y, y_pred, sample_weights, alpha):
        for i in range(len(y)):
            comparison = y == y_pred
            equal_arrays = comparison.all()
            if(equal_arrays==True):
                return sample_weights
            else:
                if  y[i]==y_pred[i]:
                    first=sample_weights[i]/2
                    expo=(1/(math.exp(2*alpha))+1)
                    sample_weights[i]=first*expo
                else:
                    first=sample_weights[i]/2
                    expo=(math.exp(2*alpha))+1
                    sample_weights[i]=first*expo
        return sample_weights

        pass

    def predict(self, X):
        out =[]
        for item in range(self.n_stumps):
            pred=self.stumps[item].predict(X)
            out.append(pred)
        predict_out = np.array(out)
        return np.sign(predict_out[0])


    def evaluate(self, X, y):
        pred = self.predict(X)
        # find correct predictions
        correct = (pred == y)

        accuracy = np.mean(correct) * 100  # accuracy calculation
        return accuracy