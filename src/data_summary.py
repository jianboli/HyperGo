# assume we already have data loaded
""" This file process the new data esprite_ord_graph.csv and use style_color as a
 product idenfier rather than khk_ean"""
import pickle
import pandas as pd
import numpy as np
# import matplotlib.pylab as plt
import scipy.sparse as sparse
class_id = 1 # 0: total data; 1: dataset 1, simple sample; 2: dataset 2, with changes
# %% load data -------------
# head data is stored in the header.txt file
data = pd.read_csv("data/raw/esprite_ord_graph.csv",
                   dtype={"khk_ean": np.str, "OUT_Items": np.int32,
                          "RET_Items": np.int32, "OrderNo": np.int32,
                          "n_subclass_code": np.str, "style_color": np.str})
data.dropna(inplace=True)
np.random.seed(1)
# %% some pre-process analysis -------
# data sample methods
if class_id == 1:
    # TODO: remove me at the production line please
    orders = data['OrderNo'].unique()
    orders = orders[np.random.permutation(orders.shape[0])[:30000]]
    data = data.iloc[np.in1d(data['OrderNo'].values, orders), :]
elif class_id == 2:
    bsk_ret_label = data.groupby("OrderNo", as_index=False).agg({"RET_Items": lambda g: sum(g) > 0})
    ret_bsk = bsk_ret_label.loc[bsk_ret_label['RET_Items'], :]
    n = ret_bsk.shape[0]
    ret_bsk = ret_bsk.iloc[np.random.permutation(n),:].iloc[:int(n*2/3), 0] # remove half of the returned baskets
    bsk_ret_label = bsk_ret_label.loc[np.logical_not(np.in1d(bsk_ret_label['OrderNo'], ret_bsk)), :]
    orders = bsk_ret_label.iloc[np.random.permutation(bsk_ret_label.shape[0])[:50000], 0].values

    data = data.loc[np.in1d(data['OrderNo'], orders), :]


def cal_ret_rate(g):
    return sum(g['RET_Items'])/sum(g['OUT_Items'])

## total data level
print(data.shape[0], cal_ret_rate(data), (data['RET_Items'] > 0).sum())

# style color level
ret_rate = data.groupby("khk_ean").agg(cal_ret_rate)
ret_rate = ret_rate[['RET_Items']]
print(ret_rate.shape[0], ret_rate['RET_Items'].mean())

# subclass level
ret_rate = data.groupby("n_subclass_code").agg(cal_ret_rate)
ret_rate = ret_rate[['RET_Items']]
print(ret_rate.shape[0], ret_rate['RET_Items'].mean())

# Order Number level
ret_rate = data.groupby("OrderNo").agg(cal_ret_rate)
ret_rate = ret_rate[['RET_Items']]
print(ret_rate.shape[0], ret_rate['RET_Items'].mean(), (ret_rate['RET_Items'] > 0).sum())

# Basket Contains Pairs
bsk_ret_label = data.groupby("OrderNo").agg({"RET_Items": lambda g: sum(g) > 0})
num_item_bsk = data.groupby(["OrderNo", "n_subclass_code"]).agg({'khk_ean': 'count'}) \
    .groupby(level='OrderNo').agg({'khk_ean': 'max'})
multi_item_bsk = num_item_bsk.query('khk_ean > 1')
multi_item_bsk_ret_rate = pd.merge(bsk_ret_label, multi_item_bsk, left_index=True, right_index=True, how='inner')
print(multi_item_bsk.shape[0], np.nan, multi_item_bsk_ret_rate['RET_Items'].sum())

# Other Baskets
single_item_bsk_ret_rate = bsk_ret_label.drop(multi_item_bsk_ret_rate.index, errors="ignore")

print(single_item_bsk_ret_rate.shape[0], np.nan, single_item_bsk_ret_rate['RET_Items'].sum())