import pandas as pd
import numpy as np
from pprint import pprint

# 5 <-- classes
# not_recom,recommend,very_recom,priority,spec_prior
# 8  <-- categories
# parents,3,usual,pretentious,great_pret  <-- attr
# has_nurs,5,proper,less_proper,improper,critical,very_crit 
# form,4,complete,completed,incomplete,foster 
# children,4,1,2,3,more 
# housing,3,convenient,less_conv,critical 
# finance,2,convenient,inconv 
# social,3,nonprob,slightly_prob,problematic 
# health,3,recommended,priority,not_recom
# 12960
# usual,proper,complete,1,convenient,convenient,nonprob,recommended,recommend



def encode_cls(f) -> dict:
    cls_map = {}
    n = int(f.readline())
    classes = f.readline().strip().split(",")
    if len(classes) != n:
        raise Exception("Invalid dataset format")
    
    for idx, cls_name in enumerate(classes):
        cls_map[cls_name] = idx

    return cls_map



def encode_attrs(f) -> dict:
    attrs_map = {}
    n = int(f.readline())
    for i in range(n):
        l = f.readline().strip()
        v, cnt, T = l.split(",", 2)
        cnt = int(cnt)
        T = T.split(",")
        if cnt != len(T):
            raise Exception("Invalid dataset format")
        attrs_map[v] = {}
        for idx, attr in enumerate(T):
            attrs_map[v][attr] = idx

    return attrs_map

def get_data(f, headers):
    print(headers)
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
    return pd.DataFrame(data=s, columns=headers)


def apply_enc(attrs_map, attr):
    return attrs_map[attr]

def apply_attr_enc(attrs_map, attr):
    one_hots = [0] * len(attrs_map)
    idx = int(attrs_map[attr])
    one_hots[idx] = 1
    one_hots = list(map(lambda x: str(x), one_hots))
    s = str.join(",", one_hots)
    return s

def encode():
    with open("nursery.data") as f:
        cls_map = encode_cls(f)
        attrs_map = encode_attrs(f)
        headers = list(attrs_map.keys())
        headers.append("cls")
        df = get_data(f, headers)
        T = df.copy()

        for i, column in enumerate(df):
            m = attrs_map.get(column, None)
            if m is not None:
                df[column] = df[column].apply(lambda x : apply_attr_enc(m, x))
            else:
                m = cls_map
                df[column] = df[column].apply(lambda x : apply_enc(m, x))

        r = df.apply(lambda x: ','.join(x.astype(str)), axis=1)

        data = r.apply(lambda x: x.split(","))
        data = data.values.tolist()
        npdata = np.array(data)
        L = len(npdata)
        l =  int(0.30 * L)

        np.random.shuffle(npdata)
        np.random.shuffle(npdata)
        mask = np.ones(npdata.shape[0],dtype=bool)
        idxs = np.random.choice(L, l, replace=False)

        mask[idxs] = False
        training_data = npdata[mask, :]
        test_data = npdata[~mask, :]


        train = pd.DataFrame(data=training_data, columns=None)
        test = pd.DataFrame(data=test_data, columns=None)

        train.to_csv("nurser-encoded-train.data", header=False, index=False)
        test.to_csv("nurser-encoded-test.data", header=False, index=False)

if __name__ == "__main__":
    encode()
    
