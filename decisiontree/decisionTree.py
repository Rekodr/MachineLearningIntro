import numpy as np
import os
import math
import json
from typing import Any, Tuple, List, Dict, NamedTuple
import time
from pprint import pprint

BASE_DIR = os.path.dirname("..")
MODEL_FILE = os.path.join(BASE_DIR, "models", "tree.json")

#TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "fishing.data")
#TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "contact-lenses.data")
TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data","Car", "car_training.data")
#TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "iris.data")
TEST_DATASET = os.path.join(BASE_DIR, "sample_data","Car", "car_test.data")


#####################################################
# @defined types
classes = List[str]
attrs = Dict[str, tuple]  # e.g: {'Wind': ('Strong', 'Weak'), 'Water': ('Warm', 'Moderate', 'Cold')}
node = Dict[str, Any]
gainsList = List[float]
maxgain = Tuple[str, int, float]
#####################################################

class DataParser():
    @classmethod
    def get_classes(cls, f) -> classes:
        n = f.readline()
        try:
            n = int(n)
        except ValueError:
            raise Exception("Invalid file format")
        
        l = f.readline().strip()
        C :classes = l.split(",")
        if len(C) != n:
            raise Exception("Invalid file format")
        return C
    
    @classmethod
    def get_variables(cls, f) -> attrs:
        n = f.readline()
        V :attrs = {}
        try:
            n = int(n)
        except ValueError:
            raise Exception("Invalid file format")

        while n > 0:
            l :str = f.readline().strip()
            v, cnt, T = l.split(",", 2)
            T = T.split(",")
            try:
                cnt = int(cnt)
            except ValueError:
                raise Exception("Invalid file format")

            if cnt != len(T):
                raise Exception("Invalid file format")

            V[v] = tuple(T)
            n -= 1

        return V
    
    @classmethod
    def get_samples(cls, f) -> np.array:
        n = f.readline()
        s = []
        try:
            n = int(n)
        except ValueError:
            raise Exception("Invalid file format")
        
        while n > 0:
            l = f.readline().strip()
            s.append(tuple(l.split(",")))
            n -= 1
        
        return np.array(s)

    @staticmethod
    def read_data(data_path: str):
        with open(data_path, "r") as f:
            C = DataParser.get_classes(f)
            V = DataParser.get_variables(f)
            D :np.array = DataParser.get_samples(f)
            return C, V, D
        f.close()

class DecisionTree():
    targets:classes = []
    attributes :attrs = {}
    trainingdata :np.array = None
    min_dataset :int = 1
    root_node :node = None
    n_random_attr = None
    validation_data = None
    test_acc = 0.0

    def __init__(self, data :np.array, attributes :attrs, targets_cls :classes, min_dataset:int=1, prune=False, n_random_attr=None, save_mdl=False):
        self.min_dataset = min_dataset
        self.trainingdata = data
        self.attributes = attributes
        self.targets = targets_cls
        self.prunne = prune
        if n_random_attr:
            self.n_random_attr = int(n_random_attr)
        self.save_mdl = save_mdl

    @staticmethod
    def setEntropy(target_attrs: classes, data: np.array) -> float:
        targets = data[:, -1]
        Y, counts = np.unique(targets, return_counts=True)
        size = len(targets)
        E = 0.0
        for y, count in zip(Y, counts):
            if y in target_attrs:
                E += -1 * (count / size) * math.log2((count / size))
        return E

    @staticmethod
    def entropy(attr_value :str, attr_idx :int, target_groups :np.array, sample_size :int) -> float:
        E = 0
        value_cnt_per_grp = []
        attr_cnt = 0
        for group in target_groups:
            lines = group[:,  attr_idx] == attr_value
            views = len(group[lines]) # cnt how many the attr values is seen
            attr_cnt += views
            value_cnt_per_grp.append(views)
        
        # calcute E for attr value
        if attr_cnt != 0:
            value_cnt_per_grp = list(map(lambda x: x/attr_cnt, value_cnt_per_grp))
            
        for p in value_cnt_per_grp: 
            if p != 0:
                E += -1 * p * math.log2(p)
        s = E * attr_cnt/sample_size
        return s

    @staticmethod
    def maxGrain(G :gainsList) -> maxgain:
        maxGain = 0.0
        attr_name = ""
        attr_idx = 0
        for idx, name, g in G:
            if g >= maxGain:
                maxGain = g
                attr_name = name
                attr_idx = idx
        return (attr_name, attr_idx, maxGain)       


    def mostCommonValue(self, data_arr :np.array) -> str:
        MC = None # most common
        max_cnt = 0
        for target in self.targets:
            t = data_arr[:, -1] == target
            r = data_arr[t]
            if len(r) > max_cnt:
                max_cnt = len(r)
                MC = target
        return MC 
    
    def createNode(attr_name :str=None, attr_idx :int=None, leaf :int=None, children :Dict[str, Any]=None) -> node:
        return {
            "aname": attr_name,
            "idx": attr_idx,
            "children": {},
            "leaf": leaf,
            "mc": None
        }

    @staticmethod
    def bestSplit(targets :classes, attributes :attrs, data :np.array, S :float, n_random_attr=None) -> maxgain:
        target_groups = []
        G = []
        for target in targets:
            t = data[:, -1] == target
            r = data[t]
            target_groups.append(r)

        N = len(data)
        A = attributes
        idxs = []
        if n_random_attr is not None and n_random_attr < len(A):
            idxs = np.random.choice(len(A), size=n_random_attr, replace=False)

        for idx, attr_name in enumerate(A):
            g = S
            for attr_value in A[attr_name]: #loop throup attr values
                E = DecisionTree.entropy(attr_value, idx, target_groups, N)
                g -= E

            if n_random_attr and n_random_attr < len(A):
                if idx in idxs:
                    G.append((idx, attr_name, g))
            else:
                G.append((idx, attr_name, g))

        attr_name, index,  max_G = DecisionTree.maxGrain(G)
        return attr_name, index, max_G

    def buildTree(self, attributes :attrs, data :np.array) -> node:
        if len(data) <= self.min_dataset: #if the number of rows is < n return most common.
            leaf = self.mostCommonValue(data)
            return DecisionTree.createNode(leaf=leaf)
        
        S = DecisionTree.setEntropy(self.targets, data)
        attr_name, attr_idx, max_gain = DecisionTree.bestSplit(self.targets, attributes, data, S, self.n_random_attr)
       
        if max_gain == 0:
            leaf = self.mostCommonValue(data)
            return DecisionTree.createNode(leaf=leaf)
        
        new_node = DecisionTree.createNode(attr_name=attr_name, attr_idx=attr_idx)
        new_attrs = attributes.copy()
        new_attrs.pop(attr_name, None)
        for attr_value in attributes[attr_name]:
            subData_idxs = data[:, attr_idx] == attr_value
            subData = data[subData_idxs]
            subData = np.delete(subData, attr_idx, axis=1) # remove traited attr
            child = self.buildTree(new_attrs, subData)
            new_node["children"][attr_value] = child
        new_node["mc"] = self.mostCommonValue(data)
        return new_node


    def traverseTree(self, currNode :node, data, mc :str=None):
        if currNode["aname"] is None:
            leaf = currNode["leaf"]
            if leaf is None:
                return mc
            return leaf
        
        attr_idx = currNode["idx"]
        value = data[attr_idx]
        next_node = currNode["children"][value]
        if next_node is None:
            raise Exception("Unkown attribute: {}".format(value))

        d = np.delete(data, attr_idx)
        return self.traverseTree(next_node, d, mc=currNode["mc"])     

    def traverseTreePruneREP(self, currNode: node, mc :str=None):
        """
        Reduced error pruning
        """
        leaf = currNode["leaf"]
        if leaf is not None:
            currNode["leaf"] = None
            acc = self.test(self.validation_data)
            if acc < self.test_acc:
                currNode["leaf"] = leaf
            else:
                self.test_acc = acc
        else:
            children = currNode["children"]
            for key in children:
                child = children[key]
                self.traverseTreePruneREP(child, mc=currNode["mc"])

    def prunneTree(self):
        # print("before pruning: {}".format(self.test_acc))
        self.traverseTreePruneREP(self.root_node)
        # print("after prunning {}".format(self.test_acc))

    def classify(self, data):
        data = np.array(data)
        predicted = self.traverseTree(self.root_node, data)
        return predicted

    def test(self, data: np.array):
        d = np.array(data)
        Y = d[:, -1]
        X = np.delete(d, -1, axis=1)
        predicted = []
        for x in X:
            predicted.append(self.classify(x))
        acc = np.mean(predicted == Y)
        return acc

    def train(self, validationData=None):
        self.root_node = self.buildTree(self.attributes, self.trainingdata)
        if validationData is not None:
            acc = self.test(validationData)
            self.test_acc = acc
            # print("adj: {}".format(self.test_acc))

        if self.prunne:
            if validationData is None:
                raise Exception("validation data is required for prunning")
            else:
                self.validation_data = validationData
                self.prunneTree()
                # print("accuracy: {}".format(acc))
        
        if self.save_mdl is True:
            with open(MODEL_FILE, 'w') as f:  
                json.dump(self.root_node, f, indent=2)
        return self.root_node

if __name__ == "__main__":
    tgt_cls, A, data = DataParser.read_data(TRAINING_DATASET)
    T, a, test_data = DataParser.read_data(TEST_DATASET)
    dt = DecisionTree(data, attributes=A, targets_cls=tgt_cls ,min_dataset=5, prune=True, n_random_attr=2)
    dt.train(validationData=test_data)
    print("accuracy: {}".format(dt.test_acc))
    # pred = dt.classify(("high","low","5","4","big","low"))
    # print(pred)