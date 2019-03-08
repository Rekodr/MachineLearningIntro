import numpy as np
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "fishing.data")
# TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "contact-lenses.data")
# TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data","Car", "car_training.data")

class Trainer():
    targets = []
    attributeMap = []
    trainingdata = None

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
    def read_data(data_path):
        with open(data_path, "r") as f:
            C = Trainer.get_classes(f)
            V = Trainer.get_variables(f)
            D :np.array = Trainer.get_samples(f)
            return C, V, D
        f.close()
                    
    @staticmethod
    def G(targets, attrs, data: np.array, S):
        target_groups = []
        G = []
        for target in targets:
            t = data[:, -1] == target
            r = data[t]
            target_groups.append(r)

        N = len(data)
        for idx, name in enumerate(attrs):
            g = S
            for value in attrs[name]:
                E = 0
                value_cnt_per_grp = []
                attr_cnt = 0
                for group in target_groups:
                    lines = group[:,  idx] == value
                    views = len(group[lines]) # cnt how many the attr values is seen
                    attr_cnt += views
                    value_cnt_per_grp.append(views)
                
                # calcute E for attr value
                value_cnt_per_grp = list(map(lambda x: x/attr_cnt, value_cnt_per_grp))
                for p in value_cnt_per_grp: 
                    if p != 0:
                        E += -1 * p * math.log2(p)
                s = E * attr_cnt/N 
                g -= s
            G.append(g)
            print("{} {}".format(name, g))
        
                

        # for idx, dim in enumerate(data[:, 0:-1].T):
        #     print("idx: {}, dim: {}".format(idx, attr[idx]))

    def S(self, data: np.array):
        targets = data[:, -1]
        Y, counts = np.unique(targets, return_counts=True)
        size = len(targets)
        E = 0
        for y, count in zip(Y, counts):
            if y in self.targets:
                E += -1 * (count / size) * math.log2((count / size))

        return E

    def train(self):
        pass


if __name__ == "__main__":
    trainer = Trainer()
    trainer.targets, trainer.attributeMap, trainer.trainingdata = Trainer.read_data(TRAINING_DATASET)
    S = trainer.S(trainer.trainingdata)
    print("dim: {}".format(trainer.attributeMap))
    Trainer.G(trainer.targets, trainer.attributeMap ,trainer.trainingdata, S)
