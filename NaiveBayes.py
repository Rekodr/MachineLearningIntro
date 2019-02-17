import pandas as pd
import numpy as np
import os
from timeit import default_timer as timer


CLASSES_LIST = [
"Atheism",
"Graphics",
"MSwindows",
"PC",
"Mac"
"Xwindows",
"Forsale",
"Autos"
"Motorcycles",
"Baseball",
"Hockey",
"Cryptology",
"Electronics"
"Medicine",
"Space",
"Christianity",
"Guns",
"MideastPolitics",
"Politics",
"Religion"
]

C = list(map(lambda c: c.lower(), CLASSES_LIST))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def read_data() -> pd.DataFrame :
    file_path = os.path.join(BASE_DIR, "sample_data", "Forum", "forumTraining.data.txt")
    D = {}
    i = 0
    V_raw = []
    with open(file_path, "r") as f:
        for line in f:
            category, text = line.split(maxsplit=1)
            document = text.split()
            V_raw.extend(document)
            D[i] = {"category": category, "document": document}
            i+=1; 
    f.close()
    df = pd.DataFrame.from_dict(D, orient="index")
    V, cnt = np.lib.arraysetops.unique(V_raw, return_counts=True)
    cnt = list(map(lambda x: 0, cnt))
    return df, dict(zip(V, cnt))



def train():
    Start = 0
    End = 0
    s = timer()
    sample, V = read_data()
    # sample["category"] = sample["category"].astype("category")
    
    # calculate sample size
    N = len(sample)
    print("Sample size: {}".format(N))
    

    pC = {} # probability of each class Cj
    nC = {} # number of words for each class Cj

    # compute prob of each class

    print("grouping")
    Start = timer()
    Docs = sample.groupby(["category"])
    for (name, docj) in Docs:
        pC[name] = len(docj) / N
    End = timer()
    print("Done in: {}".format(End - Start))

    # concatenate for each class
    print("summming")
    Start = timer()
    Textj = sample.groupby(["category"]).sum()
    End = timer()
    print("Done in: {}".format(End - Start))

    T = {} # list of word freq per category.
    T_prime = {}
    V_size = len(V)

    print("doing st")
    Start = timer()
    for category, text in Textj.itertuples():
        nC[category] = n = len(text)
        Text, f = np.lib.arraysetops.unique(text, return_counts=True)
        tmp = list(map(lambda x: (x + 1)/ (n + V_size), V.values()))
        Tb = dict(zip(list(V.keys()), tmp))
        f_prime = list(map(lambda x: (x + 1)/ (n + V_size), f))
        W = dict(zip(Text, f))
        W_ = {**Tb, **dict(zip(Text, f_prime))}
        T[category] = W
        T_prime[category] = W_
    End = timer()
    print("Done in: {}".format(End - Start))
        # print("{}: {}".format(category, nC[category]))
    
    # Table = pd.DataFrame.from_dict(T, orient="index")

    print("Creating Df")
    Start = timer()
    P = pd.DataFrame.from_dict(T_prime, orient="index")
    End = timer()
    print("Done in: {}".format(End - Start))

    # Vocab = Table.sum()
    # Pj = P.head()

    # print(Pj)
    e = timer()
    print("Elapse time: {}".format(e - s))


if __name__ == "__main__":
    train()