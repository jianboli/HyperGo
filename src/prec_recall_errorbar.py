import pandas as pd

baseline = pd.read_csv("rst/baseline.csv")
print(baseline.describe())

baseline_norm = pd.read_csv("rst/baseline_norm.csv")
print(baseline_norm.describe())

knn = pd.read_csv("rst/knn.csv")
print(knn.describe())

hypergraph = pd.read_csv("rst/hypergraph.csv")
print(hypergraph.describe())