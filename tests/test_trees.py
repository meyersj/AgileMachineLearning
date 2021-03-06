from unittest import TestCase

from sklearn.tree import DecisionTreeClassifier
from Models.trees import wrapper_for_decision_tree_in_sklearn, decision_tree_accuracy
from tests.data import DIGITS_DATASET


DATA = DIGITS_DATASET['data']
LABELS = DIGITS_DATASET['target']


class TestDecisionTrees(TestCase):

    def test_digits_solutions_with_pre_built_model(self):

        data = DATA
        labels = LABELS

        arbitrary_sample_the_default_tree_will_get_right = 3

        state_predicted_from_model = wrapper_for_decision_tree_in_sklearn(
            data, labels, data[arbitrary_sample_the_default_tree_will_get_right])

        self.assertEqual(
            labels[arbitrary_sample_the_default_tree_will_get_right],
            state_predicted_from_model,
            "Solve me first by writing `wrapper_for_decision_tree_in_sklearn`.")


    def test_minimum_accuracy_of_tree_model(self):
        """
            Split the data into a train and test set.
            Evaluate the accuracy of the model and assert it's above the
        supplied threshold.
            May require making a new function next to
        `wrapper_for_decision_tree_in_sklearn` to handle more than one test sample.

            Replace the assertion below with one that checks the model's accuracy
        is greater than or equal to the accuracy_target.
        """
        accuracy_target = 0.8  # percent
        relative_test_set_size = 0.4  # percent

        data = DATA
        labels = LABELS
        score = decision_tree_accuracy(data, labels, relative_test_set_size)
        self.assertGreaterEqual(score, accuracy_target, "score is wrong")
