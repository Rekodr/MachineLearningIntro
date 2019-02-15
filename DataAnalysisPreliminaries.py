#
# Data Analysis Preliminaries
#
import math
import matplotlib.pyplot as plt
import numpy as np

def read_data(file_name):
    """
    Read data
    @params:
        file_name: src file
    @return: file containt
    """
    content = {}
    with open(file_name) as data_file:
        for line in data_file:
            line = line.strip()
            hour, dwn = line.split(",")

            if hour == "nan" or dwn == "nan":
                # ignore data with nan values
                pass
            else:
                content[int(hour)] = int(dwn)

    X = np.array(list(content.keys()))
    Y = np.array(list(content.values()))
    return (X, Y)


def stat_analysis(X, Y):
    S_XY = 0
    S_X = 0
    S_Y = 0
    S_X2 = 0
    N = len(X)

    for x, y in zip(X, Y):
        S_XY += x * y
        S_X += x
        S_Y += y
        S_X2 += math.pow(x, 2)

    # compute slope
    Num1 = (N * S_XY) - (S_X * S_Y)
    Den1 = (N * S_X2) - (math.pow(S_X, 2))
    slope = Num1 / Den1

    # compute intercept
    Num2 = S_Y - (slope * S_X)
    intercept = Num2 / N

    print("slope: {0:.3f}, intercept: {1:.3f}".format(slope, intercept))
    return (slope, intercept)



if __name__ == "__main__":
    X, Y = read_data("sample_data/book_downloads.txt")

    font = {
        "color": "b",
        "weight": "normal",
        "size": 12,
    }
    plt.grid(True)
    plt.title("Number of Downloads for a Book on Amazon", fontdict=font)
    plt.xlabel("time(hours)", fontdict=font)
    plt.ylabel("downloads", fontdict=font)
    plt.scatter(x=X, y=Y, alpha=0.5)
    
    slope, intercept = stat_analysis(X, Y)
    Y_p = (X * slope) + intercept
    plt.plot(X, Y_p, color="r")
    plt.show()

