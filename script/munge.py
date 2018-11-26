import pandas as pd
import numpy as np
import pickle
# import matplotlib.pylab as plt
import scipy.sparse as sparse

# %% load data -------------
# head data is stored in the header.txt file
data = pd.read_csv("data/raw/esprite.csv", header=None,
                   names=["OrderDateTime", "OrderNo", "KHK_EAN", "StyleNumber",
                          "ColorNumber", "StyleSize", "OUT_Items", "RET_Items",
                          "ReturnDate", "ARTICLE_EAN", "SUBCLASS_CODE", "PRODUCT_CLASS_LONG",
                          "PRODUCT_SUBCLASS_LONG"],
                   dtype={"OrderDateTime": np.str, "OrderNo": np.int32, "KHK_EAN": np.str, "StyleNumber": np.str,
                          "ColorNumber": np.str, "StyleSize": np.str, "OUT_Items": np.int32, "RET_Items": np.int32,
                          "ReturnDate": np.str, "ARTICLE_EAN": np.str, "SUBCLASS_CODE": np.str,
                          "PRODUCT_CLASS_LONG": np.str, "PRODUCT_SUBCLASS_LONG": np.str})

data["OrderDateTime"] = pd.to_datetime(data["OrderDateTime"], infer_datetime_format=True)
data["ReturnDate"] = pd.to_datetime(data["ReturnDate"], infer_datetime_format=True)

data.to_pickle("data/clean/esprite.pkl")

# %% some pre-process analysis -------
# product return rate
data = pd.read_pickle("data/clean/esprite.pkl")

# TODO: remove me at the production line please
data = data.head(100000)


def cal_ret_rate(g): # we added 1 on both to avoid 0 return rate. This is only for the use of modeling
    return (sum(g['RET_Items'])+1)/(sum(g['OUT_Items'])+1)

ret_rate = data.groupby("KHK_EAN").agg(cal_ret_rate)
ret_rate = ret_rate[['RET_Items']]

ret_rate.to_pickle("data/clean/return_rate.pkl")

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
    return [list(g.loc[g['RET_Items']>0, "KHK_EAN"])]

bsk_ret_item_collection = data.groupby("OrderNo").agg(ret_item_collection)
bsk_ret_item_collection = bsk_ret_item_collection[['KHK_EAN']]
bsk_ret_item_collection.to_pickle("data/clean/bsk_return_item_collection.pkl")

# basket with multiple items sharing same subclass
num_item_bsk = data.groupby(["OrderNo", "SUBCLASS_CODE"]).agg({'KHK_EAN': 'count'})\
    .groupby(level='OrderNo').agg({'KHK_EAN': 'max'})
multi_item_bsk = num_item_bsk.query('KHK_EAN > 1')
multi_item_bsk_ret_rate = pd.merge(bsk_ret_label, multi_item_bsk, left_index=True, right_index=True, how='inner')

multi_item_bsk_ret_rate.to_pickle("data/clean/multi_item_bsk_return_label.pkl")

print(multi_item_bsk_ret_rate.shape)
# (1191687, 2)
print(bsk_ret_label.shape)
# (2330575, 1)
single_item_bsk_ret_rate = bsk_ret_label.drop(multi_item_bsk_ret_rate.index, errors="ignore")

print([multi_item_bsk_ret_rate['RET_Items'].mean(), bsk_ret_label['RET_Items'].mean(), single_item_bsk_ret_rate['RET_Items'].mean()])
# [0.77839147360003091, 0.60684251740450323, 0.42734052865602234]


# Construct H matrix
data['OrderNo'] = data['OrderNo'].astype('category')
data['KHK_EAN'] = data['KHK_EAN'].astype('category')
data['OrderNoIdx'] = data['OrderNo'].cat.codes
data['KHK_EANIdx'] = data['KHK_EAN'].cat.codes
bsk_item_pair = data.groupby(['OrderNoIdx', 'KHK_EANIdx']).agg({'OUT_Items': 'count', 'RET_Items' : 'sum'})


h_mat = sparse.coo_matrix((bsk_item_pair['OUT_Items'].values,
                          (bsk_item_pair.index.get_level_values("OrderNoIdx").values,
                           bsk_item_pair.index.get_level_values("KHK_EANIdx").values)))
h_mat = h_mat.tocsr()
with open("data/clean/h_mat.pkl", 'wb') as f:
    pickle.dump(h_mat, f)

r_mat = sparse.coo_matrix((bsk_item_pair['RET_Items'].values,
                           (bsk_item_pair.index.get_level_values("OrderNoIdx").values,
                            bsk_item_pair.index.get_level_values("KHK_EANIdx").values)))
r_mat = r_mat.tocsr()
with open("data/clean/r_mat.pkl", 'wb') as f:
    pickle.dump(r_mat, f)


order_no = list(data['OrderNo'].cat.categories.values)
KHK_EAN = list(data['KHK_EAN'].cat.categories.values)
with open("data/clean/order_no.pkl", 'wb') as f:
    pickle.dump(order_no, f)

with open("data/clean/KHK_EAN.pkl", 'wb') as f:
    pickle.dump(KHK_EAN, f)
