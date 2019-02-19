import pandas as pd
import numpy as np
import os
from timeit import default_timer as timer
import json
from pprint import pprint
from functools import reduce
import math

import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TRAINING SETS
TRAINING_FILE = os.path.join(BASE_DIR, "sample_data", "Forum", "forumTraining.data.txt")
TRAINING_FILE_CLEAN = os.path.join(BASE_DIR, "sample_data", "Forum", "forumTraining-clean.data.txt")

# TEST SETS
TEST_FILE = os.path.join(BASE_DIR, "sample_data", "Forum", "forumTest.data.txt")
TEST_FILE_CLEAN = os.path.join(BASE_DIR, "sample_data", "Forum", "forumTest-clean.data.txt")

# MODEL SETS
MODEL_FILE = os.path.join(BASE_DIR, "models","bayes.csv")
MODEL_FILE_CLEAN = os.path.join(BASE_DIR, "models","bayes-short.csv")



class Trainer:
    training_file_path = None
    model_file_path = None

    def __init__(self, training_file: str, model_file: str):
        self.training_file_path = training_file
        self.model_file_path = model_file
        
    def prep_training_data(self) -> tuple:
        freq = {}
        D = {}
        i = 0
        W = ""
        with open(self.training_file_path, "r") as f:
            for line in f:
                category, document = line.split(maxsplit=1)
                freq[category] = freq.get(category, 0) + 1
                D[i] = {"category": category, "document": document + " "}
                W += document + " "
                i += 1
        f.close()

        df = pd.DataFrame.from_dict(D, orient="index")
        V, cnt = np.lib.arraysetops.unique(W.split(), return_counts=True)
        cnt = list(map(lambda x: 0, cnt))    
        return df, freq, dict(zip(V, cnt))


    def words_class_prob_table(self, Docs: pd.DataFrame, V: list) -> pd.DataFrame:
        V_size = len(V)
        T = {}
        for category, doc in Docs.itertuples():
            l = doc.split()
            n = len(l)
            tmp = list(map(lambda x: (x + 1)/ (n + V_size), V.values()))
            Tb = dict(zip(list(V.keys()), tmp))
            W, f = np.lib.arraysetops.unique(doc.split(), return_counts=True)
            f_prime = list(map(lambda x: (x + 1)/ (n + V_size), f))
        
            G = {**Tb, **dict(zip(W, f_prime))}
            T[category] = G
        P = pd.DataFrame.from_dict(T, orient="index")
        return P

    def class_prob_table(self, freq: dict) -> pd.DataFrame:
        k = list(freq.keys())
        v = list(freq.values())
        total = reduce(lambda x, y: x + y, v)
        T = pd.DataFrame(data=v, index=k, columns=["prob"])
        T = T.transform(lambda x: x/total)
        return T

    def train(self):
        Start = 0
        End = 0
        Start = timer()

        sample, class_freq, V = self.prep_training_data()
        Pc = self.class_prob_table(class_freq)
        Docs = sample.groupby("category").sum()
        Pw = self.words_class_prob_table(Docs, V)
        model = pd.merge(Pc, Pw, right_index=True, left_index=True)
        pprint(model.head())
        model = model.apply(lambda x: np.log10(x))
        pprint(model.head())
        model.to_csv(self.model_file_path)

        End = timer()
        print("Elapse time: {}".format(End - Start))


class Classifier:
    test_file_path = None
    model_file_path = None
    Cp = None
    Wp = None

    dataset_size = 1

    def __init__(self, model_file, test_file):
        self.test_file_path = test_file
        self.model_file_path = model_file
        print("Loading model")
        S = timer()
        self.Cp, self.Wp = self.read_model()
        E = timer()
        print("loaded model in {}".format(E - S))

    def read_model(self) -> tuple:
        model = pd.read_csv(self.model_file_path, index_col=0).astype(float)
        model.drop(model.columns[0], axis=1, inplace=True)
        C = pd.read_csv(self.model_file_path, index_col=0, usecols=[0,1])
        return C, model

    def read_test_input(self) -> list:
        D = []
        with open(TEST_FILE, "r") as f:
            for line in f:
                category, document = line.split(maxsplit=1)
                D.append((category, document.split()))
            
        return D

    def classify(self, word_list: list) -> str:
        P = pd.DataFrame(self.Cp)
        st = timer()
        X, F =  np.lib.arraysetops.unique(word_list, return_counts=True)
        for word, cnt in zip(X, F):
            try:
                l = self.Wp[word]
                df = pd.DataFrame(l)
                df = df.apply(lambda x, cnt=cnt: x * cnt)
                P = pd.merge(P, df, right_index=True, left_index=True)
            except:
                pass
        ed = timer()
        Z = P.sum(axis=1)
        cl = Z.idxmax()
        ed = timer()

        return str(cl)

    def process_input(self, index, input_data):
        label, text = input_data
        predicted = self.classify(text)
        R = str(label) == predicted
        print ("{} / {} {}".format(index, self.dataset_size, R))
        return R

    def test(self):
        class_probs, words_probs = self.read_model()
        R = []
        data_set = self.read_test_input()
        i = 1
        self.dataset_size = len(data_set)

        print("Start testing")
        S = timer()
        results = []
        results = Parallel(n_jobs=num_cores)(delayed(self.process_input)(idx, item) for idx, item in enumerate(data_set))
        # print(results)

        E = timer()
        print("Completed in {}".format(E - S))
        s, cnt = np.lib.arraysetops.unique(results, return_counts=True)
        total = reduce(lambda x, y: x + y, cnt)
        P = list(map(lambda x: x/total, cnt))
        pprint("K: {}, V: {}".format(s, P))


if __name__ == "__main__":
    # print("Tranning")
    #trainer = Trainer(TRAINING_FILE, MODEL_FILE) 
    #trainer_clean = Trainer(TRAINING_FILE_CLEAN, MODEL_FILE_CLEAN)

    #classifier = Classifier(MODEL_FILE, TEST_FILE) 
    classifier_clean = Classifier(MODEL_FILE_CLEAN, TEST_FILE_CLEAN) 

    S = timer()
    #trainer.train()
    #trainer_clean.train()
    #classifier.test()
    classifier_clean.test()
    # print("Testing")
    # test()
    E = timer()
    print("Tested in {}".format(E - S))
