import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
from timeit import default_timer as timer
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "http://bigkix.objectsr.com:8080/alexandria-v2.1/alexandria/select"
CPCs = ["A*", "B*", "C*", "D*", "E*", "F*", "G*", "H*", "Y*"]

TRAINING_FILE = os.path.join(BASE_DIR, "sample_data", "ontologics.csv")
MAX_COUNT = 2000


def transform_abstract(obj_list:list) -> str:
    if obj_list is np.NaN:
        return np.NaN
    if obj_list is None or type(obj_list) is not list or not len(obj_list):
        return np.NaN
    soup = BeautifulSoup(obj_list[0], features="html.parser")
    tag = soup.p
    s = ""
    try:
        s = tag.string
    except Exception:
        pass
    return s

def transform_claim(obj_list:list) -> str:
    if obj_list is np.NaN:
        return np.NaN

    if obj_list is None or type(obj_list) is not list or not len(obj_list):
        return np.NaN
    return str(obj_list[0])


def apply_transformations(data: pd.DataFrame) -> pd.DataFrame:
    # Filter entries with no Abstruct
    v = pd.notna(data["ab_en"])
    data.where(v, inplace=True)

    data["ab_en"] = data["ab_en"].apply(transform_abstract)
    data["ttl_en"] = data["ttl_en"].apply(transform_claim)
    data.dropna(subset=["ab_en"], inplace=True)
    return data

# def parallelize_dataframe(df, func):
#     df_split = np.array_split(df, NUM_PARTITIONS)
#     p = Pool(NUM_CORES)
#     df = pd.concat(p.map(func, df_split))
#     p.close()
#     p.join()
#     return df


def download_doc(cpc: str, count: int) -> pd.DataFrame:
    params = dict({
        "q": "cpc:" + cpc,
        "fl": "ab_en, ttl_en",
        "rows": count
    })

    r = requests.get(url=BASE_URL, params=params)
    docs = r.json()['response']['docs']
    if docs is None:
        return

    data = pd.DataFrame(docs, columns=["ab_en", "ttl_en"])
    return data

def get_abstracts(cpc: str, count: int, cpcF: str = None):
    data = download_doc(cpc, count)
    cpc = cpc.replace("*", "")
    fn = cpc + ".csv"

    data = data.assign(cpc=cpc)
    data = apply_transformations(data) # data = parallelize_dataframe(data, apply_transformations)

    if cpcF is not None:
        cpc = cpcF
    return data



def build_trainning_set():
    df = pd.DataFrame(columns=["ab_en", "ttl_en", "cpc"])
    # start = time.time()
    for cpc in CPCs:
        print("downloading {}".format(cpc))
        d = get_abstracts(cpc, MAX_COUNT)
        df = df.append(d, ignore_index=True)
        print(df.head())
        print(df.tail())
    
    df.to_csv(TRAINING_FILE)

def train():
    S = timer()
    data = pd.read_csv(TRAINING_FILE, usecols=["cpc", "ab_en"])
    print(data.tail())
    data.drop_duplicates(inplace=True)
    classes = data.cpc.unique()
    train = data.sample(frac=0.8, random_state=32)
    test = data.drop(train.index)

    text_clf = Pipeline([
        ('vect', CountVectorizer(lowercase=True, stop_words="english")),
        ('tfidf', TfidfTransformer()),
        ('clf', ComplementNB(alpha=1)),   
    ])

    # text_clf = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
    # ])

    text_clf.fit(train["ab_en"].values.astype(str), train["cpc"].values.astype(str))
    E = timer()
    print("Completed trainning in {}".format(E - S))
    predicted = text_clf.predict(test["ab_en"].values.astype(str))
    print(metrics.classification_report(test["cpc"].values.astype(str), predicted, target_names=classes))


if __name__ == "__main__":
    #build_trainning_set()
    train()

