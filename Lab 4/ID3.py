from sklearn.datasets import load_student
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# load the cancer dataset
i = load_breast_cancer()
X = i.data
y = i.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# define the ID3 algorithm
class ID3DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        return [self.predict_row(x) for x in X]

    def predict_row(self, x):
        node = self.tree
        while node.children:
            if x[node.attribute] < node.threshold:
                node = node.children[0]
            else:
                node = node.children[1]
        return node.label

    def build_tree(self, X, y):
        if len(set(y)) == 1:
            return Node(label=y[0])
        if len(X) == 0:
            return Node(label=Counter(y).most_common(1)[0][0])
        best_attribute, best_threshold = self.find_best_split(X, y)
        left_indices = X[:, best_attribute] < best_threshold
        right_indices = X[:, best_attribute] >= best_threshold
        left_tree = self.build_tree(X[left_indices], y[left_indices])
        right_tree = self.build_tree(X[right_indices], y[right_indices])
        return Node(attribute=best_attribute, threshold=best_threshold, children=[left_tree, right_tree])

    def find_best_split(self, X, y):
        best_attribute = None
        best_threshold = None
        best_gain = -1
        for i in range(X.shape[1]):
            column = X[:, i]
            thresholds = np.unique(column)
            for j in range(len(thresholds)):
                threshold = thresholds[j]
                gain = self.information_gain(X, y, i, threshold)
                if gain > best_gain:
                    best_attribute = i
                    best_threshold = threshold
                    best_gain = gain
        return best_attribute, best_threshold

    def entropy(self, y):
        counter = Counter(y)
        probabilities = [count / len(y) for count in counter.values()]
        return -sum(p * np.log2(p) for p in probabilities)

    def information_gain(self, X, y, attribute, threshold):
        left_indices = X[:, attribute] < threshold
        right_indices = X[:, attribute] >= threshold
        left_y = y[left_indices]
        right_y = y[right_indices]
        return self.entropy(y) - (len(left_y) / len(y)) * self.entropy(left_y) - (len(right_y) / len(y)) * self.entropy(right_y)

class Node:
    def __init__(self, attribute=None, threshold=None, children=None, label=None):
        self.attribute = attribute
        self.threshold = threshold
        self.children = children
        self.label = label

# train the model
model = ID3DecisionTree()
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate the model's performance
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
