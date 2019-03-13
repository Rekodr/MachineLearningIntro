from typing import List
import numpy as np
import os
from scipy import stats
from decisionTree import DecisionTree, DataParser, attrs, classes

BASE_DIR = os.path.dirname("..")
TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data","Car", "car_training.data")
TEST_DATASET = os.path.join(BASE_DIR, "sample_data","Car", "car_test.data")


treeList = List[DecisionTree]
class RandomForest:
    trees :treeList = []
    n_trees :int = 1
    min_dataset :int = 1
    dataset :np.array = np.array([])
    attributes :attrs = {}

    def __init__(self, data :np.array, attributes :attrs, targets_cls :classes, n_trees :int=1,  min_dataset :int=1, save_mdl=False):
        self.min_dataset = min_dataset
        self.data = data
        self.attributes = attributes
        self.targets = targets_cls
        self.save_mdl = save_mdl
        self.n_trees = n_trees

    def train(self):
        for i in range(self.n_trees):
            dt :DecisionTree = DecisionTree(self.data, attributes=self.attributes, targets_cls=self.targets ,min_dataset=self.min_dataset)
            dt.train()
            self.trees.append(dt)

    def classify(self, data):
        data = np.array(data)
        predictions = []
        prediction = None
        for tree in self.trees:
            pred = tree.classify(data)
            predictions.append(pred)
        preds, cnts = np.lib.arraysetops.unique(predictions, return_counts=True)
        max_cnt = 0
        for pred, cnts in zip(preds, cnts):
            if cnts > max_cnt:
                max_cnt = cnts
                prediction = pred
        return prediction

    def test(self, data: np.array):
        d = np.array(data)
        Y = d[:, -1]
        X = np.delete(d, -1, axis=1)
        predicted = []
        for x in X:
            predicted.append(self.classify(x))
        acc = np.mean(predicted == Y)
        print("accuracy: {}".format(acc * 100))


if __name__ == "__main__":
    targets, attributes, data = DataParser.read_data(TRAINING_DATASET)
    rf = RandomForest(data, attributes, targets, n_trees=2)
    rf.train()
    rf.test(data)

