import pandas as pd
import numpy as np
import os
from timeit import default_timer as timer
import json
import multiprocessing
from joblib import Parallel, delayed
from pprint import pprint
from functools import reduce
import math

num_partitions = 10 #number of partitions to split dataframe
num_cores = 4 #number of cores on your machine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_FILE = os.path.join(BASE_DIR, "sample_data", "Forum", "forumTraining.data.txt")
TEST_FILE = os.path.join(BASE_DIR, "sample_data", "Forum", "forumTest.data.txt")
MODEL_FILE = os.path.join(BASE_DIR, "models","bayes.csv")


def prep_data():
    freq = {}
    D = {}
    i = 0
    W = ""
    with open(TRAINING_FILE, "r") as f:
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


def w_prob_table(Docs, V):
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

def c_prob_table(freq):
    k = list(freq.keys())
    v = list(freq.values())
    total = reduce(lambda x, y: x + y, v)
    T = pd.DataFrame(data=v, index=k, columns=["prob"])
    T = T.transform(lambda x: x/total)
    return T

def train():
    Start = 0
    End = 0
    Start = timer()

    sample, freq, V = prep_data()
    Pc = c_prob_table(freq)
    Docs = sample.groupby("category").sum()
    Pw = w_prob_table(Docs, V)
    model = pd.merge(Pc, Pw, right_index=True, left_index=True)
    pprint(model.head())
    model.to_csv(MODEL_FILE)

    End = timer()
    print("Elapse time: {}".format(End - Start))



def read_model() -> tuple:
    model = pd.read_csv(MODEL_FILE, index_col=0).astype(float)
    model.drop(model.columns[0], axis=1, inplace=True)
    C = pd.read_csv(MODEL_FILE, index_col=0, usecols=[0,1])
    return C, model

def get_input() -> dict:
    D = []
    with open(TEST_FILE, "r") as f:
        for line in f:
            category, document = line.split(maxsplit=1)
            D.append((category, document.split()))
        
    return D

def classify(word_list, C, W) -> str:
    P = pd.DataFrame(C)
    st = timer()
    w, f =  np.lib.arraysetops.unique(word_list, return_counts=True)
    for word, cnt in zip(w, f):
        try:
            l = W[word]
            df = pd.DataFrame(l)
            df = df.apply(lambda x: np.power(x, cnt))
            P = pd.merge(P, df, right_index=True, left_index=True)
        except:
            # print(word)
            pass
    ed = timer()
    # print("Loop workd in {}".format(ed - st))
    st = timer()
    P = P.apply(np.log10)
    Z = P.sum(axis=1)
    cl = Z.idxmax()
    ed = timer()
    # print("log in {}".format(ed - st))

    return str(cl)

def test():
    cat_probs, words_probs = read_model()
    # pprint(words_probs.head())
    R = []
    data_set = get_input()
    i = 1
    l = len(data_set)
    for k, v in data_set:
        cl = classify(v, C=cat_probs, W=words_probs)
        if(str(k) == cl):
            print("{} / {} P".format(i, l))
            R.append(True)
        else:
            print("{} / {} N".format(i, l))
            R.append(False)
        i+=1
    
    s, cnt = np.lib.arraysetops.unique(R, return_counts=True)
    total = reduce(lambda x, y: x + y, cnt)
    # pprint(total)
    P = list(map(lambda x: x/total, cnt))
    pprint("K: {}, V: {}".format(s, P))


if __name__ == "__main__":
    # print("Tranning")
    # train()
    S = timer()
    print("Testing")
    test()
    E = timer()
    print("Tested in {}".format(E - S))