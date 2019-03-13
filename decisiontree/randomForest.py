from typing import List
import numpy as np
import os
import math
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
            dt :DecisionTree = DecisionTree(self.data, attributes=self.attributes, targets_cls=self.targets 
                ,min_dataset=self.min_dataset, n_random_attr=math.sqrt(len(self.attributes)))
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
        print("N: {} cut: {} cut: {}".format(self.n_trees, self.min_dataset ,acc * 100))
        return acc


def train_loop(min_ntrees, max_ntrees):
    max_acc = 0.0
    n = 0
    cut = 1
    targets, attributes, data = DataParser.read_data(TRAINING_DATASET)
    T, a, test_data = DataParser.read_data(TEST_DATASET)

    for i in range(min_ntrees, max_ntrees + 1):
        for ncut in range(1, len(attributes) + 1):
            for j in range(0, 10):
                rf = RandomForest(data, attributes, targets, n_trees=i, min_dataset=ncut)
                rf.train()
                acc = rf.test(test_data)
                if acc > max_acc:
                    max_acc = acc
                    n = i
                    cut = ncut
    
    print("Best N: {}, cut: {}, acc: {}".format(n, cut, max_acc))


if __name__ == "__main__":
    targets, attributes, data = DataParser.read_data(TRAINING_DATASET)
    T, a, test_data = DataParser.read_data(TEST_DATASET)
    rf = RandomForest(data, attributes, targets, n_trees=13, min_dataset=5)
    rf.train()
    rf.test(data)
    rf.test(test_data)
    #train_loop(1, 2)
