from typing import List
import numpy as np
import pandas as pd
import os
import math
import multiprocessing
from joblib import Parallel, delayed
from decisionTree import DecisionTree, DataParser, attrs, classes

num_cores = multiprocessing.cpu_count()

BASE_DIR = os.path.dirname("..")
#TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data","Car", "car_training.data")
TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "nursery.data")
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

    def train(self, validationData=None):
        self.trees = []
        n = math.sqrt(len(self.attributes))
        L = len(self.data)
        l =  int(0.49 * L)
        for i in range(self.n_trees):
            np.random.shuffle(self.data)
            idxs = np.random.choice(L, l, replace=True)
            sample = self.data[idxs, :]
            dt :DecisionTree = DecisionTree(sample, attributes=self.attributes, targets_cls=self.targets 
                ,min_dataset=self.min_dataset, n_random_attr=n, prune=False)
            dt.train(validationData=validationData)
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
        return acc


def processConfig(data, test_data, attributes, targets, i):
    R = []
    d = np.array(data)
    for ncut in range(1, len(attributes) + 8):
        for j in range(0, 5):
            rf = RandomForest(d, attributes, targets, n_trees=i, min_dataset=ncut)
            rf.train()
            acc = rf.test(test_data)
            R.append([i, ncut, acc * 100])
            print("N: {} cut: {} acc: {}".format(i, ncut ,acc * 100))

    return R

def train_loop(min_ntrees, max_ntrees):
   targets, attributes, data = DataParser.read_data(TRAINING_DATASET)
   T, a, test_data = DataParser.read_data(TEST_DATASET)
#    data = np.concatenate((data, test_data))
   L = len(data)
   l =  int(0.30 * L)
   n = int(math.sqrt(len(attributes)))
   for i in range(0, 20):
    mask = np.ones(data.shape[0],dtype=bool)
    idxs = np.random.choice(L, l, replace=False)
    mask[idxs] = False
    training_data = data[mask, :]
    test_data = data[~mask, :]
    R = []
    r = Parallel(n_jobs=num_cores)(delayed(processConfig)(training_data, test_data, attributes, targets, i) for i in range(min_ntrees, max_ntrees + 1, 2) )
    for x in r:
        R += x
    
    df = pd.DataFrame(data=np.array(R), columns=["trees", "split_attr", "accuracy"])
    df = df.astype({"trees": int, "split_attr": int})
    df.to_csv("resultsNursing.csv", index=False)


if __name__ == "__main__":
    train_loop(1, 50)
