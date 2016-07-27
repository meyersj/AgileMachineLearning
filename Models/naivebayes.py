from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import preprocessing

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

