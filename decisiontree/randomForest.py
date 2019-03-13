from typing import List
import numpy as np
import os
from decisionTree import DecisionTree

BASE_DIR = os.path.dirname("..")
TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data","Car", "car_training.data")
TEST_DATASET = os.path.join(BASE_DIR, "sample_data","Car", "car_test.data")


treeList = List[DecisionTree]
class RandomForest:
    trees :treeList = []
    n_trees :int = 1
    min_dataset :int = 1
    dataset :np.array = np.array([])

    def __init__(self, dataset :np.array, n_trees=1, min_dataset=1):
        self.n_trees = n_trees
        self.min_dataset = min_dataset

    def train(self):
        for i in range(self.n_trees):
            pass

if __name__ == "__main__":
    targets, attrs, data = DecisionTree.read_data(TRAINING_DATASET)
    rf = RandomForest(data, n_trees=1)
    rf.train()

