import numpy as np
import pandas
from pprint import pprint


files = ["digits-training.data", "digits-test.data"]
files_out = ["handWrittenDigitNormalized-training.data", "handWrittenDigitNormalized-testing.data"]
for f_name, f_out in zip(files, files_out):
    with open(f_name, "r") as f:
        X_ = []
        Y_ = []
        for line in f:
            c  = line.split(" ")
            c = list(map(lambda x: int(x), c))
            X_.append(c[0:-1])
            Y_.append(c[-1])

    X = np.array(X_)
    Y = np.array(Y_)
    X = X/16
    row, col = X.shape
    arr = np.insert(X, col, Y, axis=1)
    print(arr.shape)
    df = pandas.DataFrame(arr)
    df.to_csv(f_out, index=False, header=False)

    f.close()