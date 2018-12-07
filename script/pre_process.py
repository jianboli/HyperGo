import pickle
import numpy as np
import pandas as pd
#from src.hyper_graph import HyperGraph

# constants ==========================================
train_rate = 0.6
validate_rate = 0.2
test_rate = 1-train_rate-validate_rate

# Read data ==========================================
with open("../data/order_no.pkl", 'rb') as f:
    order_no = pickle.load(f)
with open("../data/style_color.pkl", 'rb') as f:
    khk_ean = pickle.load(f)
with open("../data/h_mat.pkl", 'rb') as f:
    h = pickle.load(f)
with open("../data/r_mat.pkl", 'rb') as f:
    r = pickle.load(f)

#bsk_label = pd.read_pickle("../data/bsk_return_label.pkl")
#return_rate = pd.read_pickle("../data/return_adj_rate.pkl")
bsk_label = pd.DataFrame(r.sum(axis=1)>0, index=order_no, columns=['RET_Items'])
return_rate = pd.DataFrame(((r.sum(axis=0)+1)/(h.sum(axis=0)+1)).T, index=khk_ean, columns=['RET_Items'])



return_rate = return_rate.loc[khk_ean, :]
bsk_label = bsk_label.loc[order_no, :]

# split train test sets ===============================
m = len(order_no)
np.random.seed(1)
rnd_idx = np.random.permutation(m)
split_pt = int(m*train_rate)
train_idx = rnd_idx[:split_pt]
test_idx = rnd_idx[split_pt:]

# train
order_no_train = [order_no[i] for i in train_idx]
h_train = h[train_idx, :]
r_train = r[train_idx, :]
bsk_label_train = bsk_label.iloc[train_idx, :]

# remove the product that did not appear in train set
zero_edges = h_train.sum(0).A1 == 0
non_zero_edges = np.where(np.logical_not(zero_edges))[0]
h_train = h_train[:, non_zero_edges]
r_train = r_train[:, non_zero_edges]
khk_ean_train = [khk_ean[i] for i in non_zero_edges]
return_rate_train = return_rate.iloc[non_zero_edges, :]

# validate & test

h_test_tmp = h[test_idx, :][:, zero_edges]
idx = h_test_tmp.sum(1).A1 == 0
test_idx = test_idx[idx]  # only the baskets has all the products that appeared in train
                          # removed 11% of the test data

h_test = h[test_idx, :][:, non_zero_edges]
r_test = r[test_idx, :][:, non_zero_edges]
order_no_test = [order_no[i] for i in test_idx]
bsk_label_test = bsk_label.iloc[test_idx, :]

# test validate split
split_pt = int(len(order_no_test) * (validate_rate/(validate_rate + test_rate)))
h_validate = h_test[:split_pt, :]
h_test = h_test[split_pt:, :]

r_validate = r_test[:split_pt, :]
r_test = r_test[split_pt:, :]

order_no_validate = order_no_test[:split_pt]
order_no_test = order_no_test[split_pt:]

bsk_label_validate = bsk_label_test[:split_pt]
bsk_label_test = bsk_label_test[split_pt:]

# write out the files

with open("../data/order_no_train.pkl", 'wb') as f:
    pickle.dump(order_no_train, f)
with open("../data/khk_ean_train.pkl", 'wb') as f:
    pickle.dump(khk_ean_train, f)
with open("../data/bsk_label_train.pkl", 'wb') as f:
    pickle.dump(bsk_label_train, f)
with open("../data/return_rate_train.pkl", 'wb') as f:
    pickle.dump(return_rate_train, f)
with open("../data/h_train.pkl", 'wb') as f:
    pickle.dump(h_train, f)
with open("../data/r_train.pkl", 'wb') as f:
    pickle.dump(r_train, f)


with open("../data/h_validate.pkl", 'wb') as f:
    pickle.dump(h_validate, f)
with open("../data/r_validate.pkl", 'wb') as f:
    pickle.dump(r_validate, f)
with open("../data/order_no_validate.pkl", 'wb') as f:
    pickle.dump(order_no_validate, f)
with open("../data/bsk_label_validate.pkl", 'wb') as f:
    pickle.dump(bsk_label_validate, f)

with open("../data/h_test.pkl", 'wb') as f:
    pickle.dump(h_test, f)
with open("../data/r_test.pkl", 'wb') as f:
    pickle.dump(r_test, f)
with open("../data/order_no_test.pkl", 'wb') as f:
    pickle.dump(order_no_test, f)
with open("../data/bsk_label_test.pkl", 'wb') as f:
    pickle.dump(bsk_label_test, f)
