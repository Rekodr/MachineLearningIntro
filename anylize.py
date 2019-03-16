import pandas as pd

dt1 :pd.DataFrame = pd.read_csv("results.csv") # +
dt2 :pd.DataFrame = pd.read_csv("resultsTree.csv") # -
dt3 :pd.DataFrame = pd.read_csv("resultsTree2.csv") # -
dt5 :pd.DataFrame = pd.read_csv("t.csv") # +
dt6 :pd.DataFrame = pd.read_csv("resultsTree4.csv") # nursing
dt7 :pd.DataFrame = pd.read_csv("resultsNursing.csv")
# print(dt2.describe())
# print(dt3.describe())
# print(dt5.describe())

# print(dt6.describe())
# print(dt7.describe())

df = dt6
print(df.describe())
idx=df.groupby(by=['cut', "split_attr", "pruned"])['accuracy'].idxmax()
#idx=df.groupby(by=['trees', "split_attr"])['accuracy'].idxmax()
df_max=df.loc[idx,]
print("*" * 20)
print(df_max)
# df_max.to_csv("rf_analysis.csv", index=False)
