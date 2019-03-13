import numpy as np
import os
import math
import json
from typing import Tuple, List, Dict

BASE_DIR = os.path.dirname("..")
MODEL_FILE = os.path.join(BASE_DIR, "models", "tree.json")

TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "fishing.data")
#TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "contact-lenses.data")
#TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data","Car", "car_training.data")
#TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "iris.data")
TEST_DATASET = os.path.join(BASE_DIR, "sample_data","Car", "car_test.data")


class DecisionTree():
    targets = []
    attributes = []
    trainingdata = None
    root_node = None
    min_dataset = 1

    def __init__(self, min_dataset:int=1):
        self.min_dataset = min_dataset

    @staticmethod
    def get_classes(f):
        n = f.readline()
        try:
            n = int(n)
        except ValueError:
            raise Exception("Invalid file format")
        
        l = f.readline().strip()
        C = l.split(",")
        if len(C) != n:
            raise Exception("Invalid file format")
        return C
    
    @staticmethod
    def get_variables(f):
        n = f.readline()
        V = {}
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
    
    @staticmethod
    def get_samples(f) -> np.array:
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
            C = DecisionTree.get_classes(f)
            V = DecisionTree.get_variables(f)
            D :np.array = DecisionTree.get_samples(f)
            return C, V, D
        f.close()

    @staticmethod
    def setEntropy(target_attrs, data: np.array) -> float:
        targets = data[:, -1]
        Y, counts = np.unique(targets, return_counts=True)
        size = len(targets)
        E = 0.0
        for y, count in zip(Y, counts):
            if y in target_attrs:
                E += -1 * (count / size) * math.log2((count / size))
        return E

    @staticmethod
    def entropy(attr_value:str, attr_idx:int, target_groups, sample_size:int):
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

    #####################################################
    # @defined types
    gainsList = List[float]
    maxgain = Tuple[str, int, float]
    #####################################################
    @staticmethod
    def maxGrain(G:gainsList) -> maxgain:
        maxGain = 0.0
        attr_name = ""
        attr_idx = 0
        for idx, name, g in G:
            if g >= maxGain:
                maxGain = g
                attr_name = name
                attr_idx = idx
        return (attr_name, attr_idx, maxGain)       

    @staticmethod
    def bestSplit(targets, attrs:dict, data:np.array, S:float) -> maxgain:
        target_groups = []
        G = []
        for target in targets:
            t = data[:, -1] == target
            r = data[t]
            target_groups.append(r)

        N = len(data)
        for idx, attr_name in enumerate(attrs):
            g = S
            for attr_value in attrs[attr_name]: #loop throup attr values
                E = DecisionTree.entropy(attr_value, idx, target_groups, N)
                g -= E
            G.append((idx, attr_name, g))

        attr_name, index,  max_G = DecisionTree.maxGrain(G)
        return attr_name, index, max_G

    def most_common(self, data_arr:np.array) -> str:
        MC = None # most common
        max_cnt = 0
        for target in self.targets:
            t = data_arr[:, -1] == target
            r = data_arr[t]
            if len(r) > max_cnt:
                max_cnt = len(r)
                MC = target
        return MC 
    
    #####################################################
    # @defined types
    attrs = Dict[str, tuple]  # e.g: {'Wind': ('Strong', 'Weak'), 'Water': ('Warm', 'Moderate', 'Cold')}
    #####################################################    
    def buildTree(self, attributes:attrs, data_arr: np.array, parent_node:dict = None, edge_name:str=None):
        if len(data_arr) <= self.min_dataset: #if the number of rows is < n return most common.
            most_common = self.most_common(data_arr) 
            parent_node["+"][edge_name] = {"$": most_common, "@": None}
            return 
        
        S = DecisionTree.setEntropy(self.targets, data_arr)
        attr_name, attr_idx, max_gain = DecisionTree.bestSplit(self.targets, attributes, data_arr, S)
       
        new_node = None
        if max_gain == 0:
            most_common = self.most_common(data_arr)
            parent_node["+"][edge_name] = {"$": most_common, "@": None}
            return
        else:
            new_node = {
                "@": attr_name,
                "idx": attr_idx,
                "+": {}
            }
            if parent_node is None:
                self.root_node = new_node
            else:
                parent_node["+"][edge_name] = new_node

        new_attrs = attributes.copy()
        new_attrs.pop(attr_name, None)
        for attr_value in attributes[attr_name]:
            data = data_arr[:, attr_idx] == attr_value
            data = data_arr[data]
            data = np.delete(data, attr_idx, axis=1)
            self.buildTree(new_attrs, data, parent_node=new_node, edge_name=attr_value)

    def train(self):
        self.buildTree(self.attributes, self.trainingdata)
        return self.root_node

    def traverseTree(self, node, data):
        if node["@"] is None:
            return node["$"]
        
        attr_idx = node["idx"]
        value = data[attr_idx]
        next_node = node["+"][value]
        d = np.delete(data, attr_idx)
        return self.traverseTree(next_node, d)     

    def classify(self, data):
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
        print("accuracy: {}".format(acc * 100))



if __name__ == "__main__":
    trainer = DecisionTree(min_dataset=1)
    trainer.targets, trainer.attributes, trainer.trainingdata = DecisionTree.read_data(TRAINING_DATASET)
    trainer.train()
    T, a, test_data = DecisionTree.read_data(TEST_DATASET)
    trainer.test(trainer.trainingdata)

    # with open(MODEL_FILE, 'w') as f:  
    #     json.dump(trainer.root_node, f, indent=2)