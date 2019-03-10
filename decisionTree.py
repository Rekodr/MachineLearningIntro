import numpy as np
import os
import math
import json
from pprint import pprint


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "models", "tree.json")

TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "fishing.data")
#TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "contact-lenses.data")
#TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data","Car", "car_training.data")


class DecisionTree():
    targets = []
    attributes = []
    trainingdata = None
    root_node = None

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
    def S(target_attrs, data: np.array):
        targets = data[:, -1]
        Y, counts = np.unique(targets, return_counts=True)
        size = len(targets)
        E = 0
        for y, count in zip(Y, counts):
            if y in target_attrs:
                E += -1 * (count / size) * math.log2((count / size))
        return E

    @staticmethod
    def Entropy(attr_value, attr_idx, target_groups, sample_size):
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
    def max_gain(G):
        max_G = 0
        attr_name = ""
        attr_idx = 0
        # print("g: {}".format(G))
        for idx, name, g in G:
            if g >= max_G:
                max_G = g
                attr_name = name
                attr_idx = idx
        return (attr_name, attr_idx, max_G)       

    @staticmethod
    def max_G(targets, attrs:dict, data:np.array, S:float):
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
                E = DecisionTree.Entropy(attr_value, idx, target_groups, N)
                g -= E
            G.append((idx, attr_name, g))

        attr_name, index,  max_G = DecisionTree.max_gain(G)
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
    
    def build_tree(self, attributes:dict, data_arr: np.array, parent_node:dict = None, edge_name:str=None):
        if len(data_arr) <= 1: #if the number of rows is < n return most common.
            most_common = self.most_common(data_arr) 
            parent_node["+"][edge_name] = {"$": most_common, "@": None}
            # print("{} -> {} : {}".format(parent, edge, most_common))         
            return 
        
        S = DecisionTree.S(self.targets, data_arr)
        attr_name, attr_idx, max_gain = DecisionTree.max_G(self.targets, attributes, data_arr, S)
       
        new_node = None
        if max_gain == 0:
            most_common = self.most_common(data_arr)
            parent_node["+"][edge_name] = {"$": most_common, "@": None}
            # print("{} -> {} : {}".format(parent, edge, most_common))
            return
        else:
            new_node = {
                "@": attr_name,
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
            self.build_tree(new_attrs, data, parent_node=new_node, edge_name=attr_value)



    def train(self):
        self.build_tree(self.attributes, self.trainingdata)


    def test(self, training_data: np.array):
        pass
if __name__ == "__main__":
    trainer = DecisionTree()
    trainer.targets, trainer.attributes, trainer.trainingdata = DecisionTree.read_data(TRAINING_DATASET)
    trainer.train()

    with open(MODEL_FILE, 'w') as f:  
        json.dump(trainer.root_node, f, indent=2)