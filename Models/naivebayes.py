from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import preprocessing
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder

NB = MultinomialNB
#NB = GaussianNB

datakey = {
    0: {"Near":0, "None":1, "Urgent":2},
    1: {"Yes":3, "No":4},
    2: {"Yes":3, "No":4}
}

labelkey = {
    "Party": 5,
    "Pub": 6,
    "Study": 7,
    "TV": 8
}

rlabelkey = {
    5: "Party",
    6: "Pub",
    7: "Study",
    8: "TV"
}


def wrapper_for_nb_in_sklearn(data, current_state_to_predict):
    """
        Import an already-built implementation, train it on the data,
    and return the class predicted given the current state.

        Note that the last column of data is assumed to be the variable
    to predict, and the order
    """
    """
    factors = [x[0:3] for x in data]
    states = [x[3] for x in data]

    # convert data values into integers using datakey lookup
    dataset = []
    for i, row in enumerate(factors):
        dataset.append([])
        for j, value in enumerate(row):
            dataset[i].append(datakey[j][value])

    #print factors
    #print dataset

    # convert labels into integers using labelkey
    labels = []
    for state in states:
        labels.append(labelkey[state])

    #print
    #print states
    #print labels

    # convert predictions to integers using datakey
    predicts = []
    for i, pred in enumerate(current_state_to_predict):
        predicts.append(datakey[i][pred])

    print dataset
    # fit data and labels to model
    clf = NB()
    clf.fit(dataset, labels)
    pred = clf.predict(predicts)

    print clf.predict_proba(predicts)
    # convert prediction back to string
    return rlabelkey[pred[0]]

    """

    # Convert inputs to arrays to leverage numpy's reshaping and indexing
    data = np.array(data)
    state_to_predict = np.array(current_state_to_predict).reshape((1, -1))

    # Convert strs to ints for all Inputs:
    le = LabelEncoder()
    all_states = data.flatten()
    le.fit(all_states)

    intified_data = le.transform(data)
    intified_state_to_predict = le.transform(state_to_predict)

    # Pick out X and y from data:
    X = intified_data[:, :-1]
    y = data[:, -1]

    # Create and train model:
    clf = MultinomialNB(alpha=1., fit_prior=True,)
    clf.fit(X, y)

    # Predict for sample, and convert back to string:
    predicted_state_as_str = clf.predict(intified_state_to_predict)[0]

    return predicted_state_as_str




datakey = {
    "Yes":0,
    "No":1
}

labelkey = {
    "Party": 3,
    "Pub": 4,
    "Study": 5,
    "TV": 6
}

rlabelkey = {
    3: "Party",
    4: "Pub",
    5: "Study",
    6: "TV"
}

def wrapper_for_nb_in_sklearn_using_bernoulli_nb(data, current_state_to_predict):
    """
        Import an already-built implementation, train it on the data,
    and return the class predicted given the current state.

        Note that the last column of data is assumed to be the variable
    to predict, and the order
    """
    # split features and labels
    factors = [x[0:5] for x in data]
    states = [x[5] for x in data]

    # convert data values into integers using datakey lookup
    features = []
    for i, row in enumerate(factors):
        features.append([])
        for j, value in enumerate(row):
            features[i].append(datakey[value])

    # convert labels into integers using labelkey
    labels = []
    for state in states:
        labels.append(labelkey[state])

    # convert predictions to integers using datakey
    predicts = []
    for i, pred in enumerate(current_state_to_predict):
        predicts.append(datakey[pred])

    # fit data and labels to model
    clf = BernoulliNB()
    clf.fit(features, labels)
    pred = clf.predict(predicts)

    # convert prediction back to string
    return rlabelkey[pred[0]]
