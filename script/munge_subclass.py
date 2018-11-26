""" This file process the new data esprite_ord_graph.csv and use style_color as a
 product idenfier rather than khk_ean"""
import pickle
import pandas as pd
import numpy as np
# import matplotlib.pylab as plt
import scipy.sparse as sparse

# %% load data -------------
# head data is stored in the header.txt file
data = pd.read_csv("data/raw/esprite_ord_graph.csv",
                   dtype={"khk_ean": np.str, "OUT_Items": np.int32,
                          "RET_Items": np.int32, "OrderNo": np.int32,
                          "n_subclass_code": np.str, "style_color": np.str})
data.dropna(inplace=True)
np.random.seed(1)
# %% some pre-process analysis -------
# product return rate

# TODO: remove me at the production line please
orders = data['OrderNo'].unique()
orders = orders[np.random.permutation(orders.shape[0])[:60000]]
data = data.iloc[np.in1d(data['OrderNo'].values, orders), :]


def cal_ret_rate(g):
    return sum(g['RET_Items'])/sum(g['OUT_Items'])

# we added 1 on both to avoid 0 return rate. This is only for the use of modeling
def cal_adj_ret_rate(g):
    return (sum(g['RET_Items'])+1)/(sum(g['OUT_Items'])+1)

ret_rate = data.groupby("style_color").agg(cal_adj_ret_rate)
ret_rate = ret_rate[['RET_Items']]

ret_rate.to_pickle("data/clean/return_adj_rate.pkl")

# basket return rate
bsk_ret_rate = data.groupby("OrderNo").agg(cal_ret_rate)
bsk_ret_rate = bsk_ret_rate[['RET_Items']]
bsk_ret_rate.to_pickle("data/clean/bsk_return_rate.pkl")

# basket return label: if the basket contains item that is returned
bsk_ret_label = data.groupby("OrderNo").agg({"RET_Items": lambda g: sum(g) > 0})
#bsk_ret_rate = ret_rate[['RET_Items']]

bsk_ret_label.to_pickle("data/clean/bsk_return_label.pkl")

# basket return item collection
def ret_item_collection(g):
    return [list(g.loc[g['RET_Items']>0, "style_color"])]

bsk_ret_item_collection = data.groupby("OrderNo").agg(ret_item_collection)
bsk_ret_item_collection = bsk_ret_item_collection[['style_color']]
bsk_ret_item_collection.to_pickle("data/clean/bsk_return_item_collection.pkl")

# basket with multiple items sharing same subclass
num_item_bsk = data.groupby(["OrderNo", "n_subclass_code"]).agg({'style_color': 'count'})\
    .groupby(level='OrderNo').agg({'style_color': 'max'})
multi_item_bsk = num_item_bsk.query('style_color > 1')
multi_item_bsk_ret_rate = pd.merge(bsk_ret_label, multi_item_bsk, left_index=True, right_index=True, how='inner')

multi_item_bsk_ret_rate.to_pickle("data/clean/multi_item_bsk_return_label.pkl")

print(multi_item_bsk_ret_rate.shape)
# (1764285, 2)
print(bsk_ret_label.shape)
# (3616144, 1)
single_item_bsk_ret_rate = bsk_ret_label.drop(multi_item_bsk_ret_rate.index, errors="ignore")

print([multi_item_bsk_ret_rate['RET_Items'].mean(), bsk_ret_label['RET_Items'].mean(), single_item_bsk_ret_rate['RET_Items'].mean()])
# [0.81487741493012755, 0.64546848798056711, 0.48407087148643607]

# # basket with multiple items sharing same subclass (under original definition)
# data['SubClassCode'] = data['n_subclass_code'].map(lambda x: x[-3:])
# num_item_bsk = data.groupby(["OrderNo", "SubClassCode"]).agg({'style_color': 'count'})\
#     .groupby(level='OrderNo').agg({'style_color': 'max'})
# multi_item_bsk = num_item_bsk.query('style_color > 1')
# multi_item_bsk_ret_rate = pd.merge(bsk_ret_label, multi_item_bsk, left_index=True, right_index=True, how='inner')
#
# print(multi_item_bsk_ret_rate.shape)
# # (2183808, 2)
# print(bsk_ret_label.shape)
# # (3616144, 1)
# single_item_bsk_ret_rate = bsk_ret_label.drop(multi_item_bsk_ret_rate.index, errors="ignore")
# print([multi_item_bsk_ret_rate['RET_Items'].mean(), bsk_ret_label['RET_Items'].mean(), single_item_bsk_ret_rate['RET_Items'].mean()])
# [0.78720610969462512, 0.64546848798056711, 0.4293685280548698]

# Construct H matrix
print(data.shape)
#data = data.loc[np.in1d(data['OrderNo'].values, multi_item_bsk.index.values), :]
#print(data.shape)
data['OrderNo'] = data['OrderNo'].astype('category')
data['style_color'] = data['style_color'].astype('category')
data['OrderNoIdx'] = data['OrderNo'].cat.codes
data['style_colorIdx'] = data['style_color'].cat.codes
data['RET_Items'] = np.sign(data['RET_Items'].values)
bsk_item_pair = data.groupby(['OrderNoIdx', 'style_colorIdx']).agg({'OUT_Items': 'count', 'RET_Items': 'sum'})


h_mat = sparse.coo_matrix((bsk_item_pair['OUT_Items'].values,
                          (bsk_item_pair.index.get_level_values("OrderNoIdx").values,
                           bsk_item_pair.index.get_level_values("style_colorIdx").values)))
h_mat = h_mat.tocsr()
with open("data/clean/h_mat.pkl", 'wb') as f:
    pickle.dump(h_mat, f)

r_mat = sparse.coo_matrix((bsk_item_pair['RET_Items'].values,
                           (bsk_item_pair.index.get_level_values("OrderNoIdx").values,
                            bsk_item_pair.index.get_level_values("style_colorIdx").values)))
r_mat = r_mat.tocsr()
with open("data/clean/r_mat.pkl", 'wb') as f:
    pickle.dump(r_mat, f)


order_no = list(data['OrderNo'].cat.categories.values)
style_color = list(data['style_color'].cat.categories.values)
with open("data/clean/order_no.pkl", 'wb') as f:
    pickle.dump(order_no, f)

with open("data/clean/style_color.pkl", 'wb') as f:
    pickle.dump(style_color, f)

