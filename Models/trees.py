import numpy as np
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score



def split_data(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def build_tree(X, y):
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf


def decision_tree_accuracy(X, y, test_size):
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    clf = build_tree(X_train, y_train)
    pred = clf.predict(X_test)
    return accuracy_score(pred, y_test)


def wrapper_for_decision_tree_in_sklearn(X, y, current_state_to_predict):
    clf = build_tree(X, y)
    pred = clf.predict(current_state_to_predict)
    predicted_state = pred
    return predicted_state
