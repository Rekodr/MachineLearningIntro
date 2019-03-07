import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATASET = os.path.join(BASE_DIR, "sample_data", "Car","car_training.data")

class Trainer():
    classes = []
    attributeMap = {}
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

    def read_data(self, data_path):
        with open(data_path, "r") as f:
            C = Trainer.get_classes(f)
            V = Trainer.get_variables(f)
            S :np.array = Trainer.get_samples(f)
            print(S)
        f.close()
                    
                        
    def train(self):
        pass


if __name__ == "__main__":
    trainer = Trainer()
    trainer.read_data(TRAINING_DATASET)
