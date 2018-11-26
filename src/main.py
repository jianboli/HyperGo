import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn import metrics
# from scoop import futures

from src.hyper_graph import HyperGraph
from src.nibble import nibble
# constants ==========================================
# parallel 0: single thread
#          1: multiprocessing package
#          2: scoop package
parallel = 1

max_num_test = 1000

n_cpu = 7
chunk_size = 1
bs = [13] #np.linspace(9, 9, 1)
phis = [0.6] #np.linspace(0.4, 0.8, 1)

test_set = "validate"
# Load data ==========================================
with open("data/clean/order_no_train.pkl", 'rb') as f:
    order_no_train = pickle.load(f)
with open("data/clean/khk_ean_train.pkl", 'rb') as f:
    khk_ean_train = pickle.load(f)
bsk_label_train = pd.read_pickle("data/clean/bsk_label_train.pkl")
return_rate_train = pd.read_pickle("data/clean/return_rate_train.pkl")
with open("data/clean/h_train.pkl", 'rb') as f:
    h_train = pickle.load(f)
with open("data/clean/r_train.pkl", 'rb') as f:
    r_train = pickle.load(f)

with open("data/clean/h_"+ test_set +".pkl", 'rb') as f:
    h_test = pickle.load(f)
with open("data/clean/r_"+ test_set +".pkl", 'rb') as f:
    r_test = pickle.load(f)
with open("data/clean/order_no_"+ test_set +".pkl", 'rb') as f:
    order_no_test = pickle.load(f)
bsk_label_test = pd.read_pickle("data/clean/bsk_label_"+ test_set +".pkl")
multi_item_bsk = pd.read_pickle("data/clean/multi_item_bsk_return_label.pkl")

bsk_ret_item_collection = pd.read_pickle("data/clean/bsk_return_item_collection.pkl")
# Construct graph ====================================
multi_idx = np.in1d(bsk_label_train.index.values, multi_item_bsk.index.values)
ratio = bsk_label_train[multi_idx].sum()[0]/multi_idx.sum() /\
        (bsk_label_train[np.logical_not(multi_idx)].sum()[0]/np.logical_not(multi_idx).sum())
# ratio = 3

bsk_label_train.loc[multi_idx, 'RET_Items'] = bsk_label_train.loc[multi_idx, 'RET_Items']/ratio

multi_idx = np.in1d(bsk_label_test.index.values, multi_item_bsk.index.values)
bsk_label_test.loc[multi_idx, 'RET_Items'] = bsk_label_test.loc[multi_idx, 'RET_Items']/ratio

g = HyperGraph(order_no_train, khk_ean_train,
               return_rate_train['RET_Items'].values,
               h_train, bsk_label_train['RET_Items'].values,
               r_train)

# For each new basket, insert into the graph and predict based on its neighbors ===
def eval_i(args):
    g_i = args[0]
    b = args[1]
    phi = args[2]
    last_idx = len(g_i.vertex_name)-1
    res = nibble(g_i, last_idx, b, phi)

    res = res[res != last_idx]
    if(len(res) != 0):
        pred = (g_i.get_labels(res).sum()*0.5 +
                (g_i.r_mat[res,:].dot(g_i.h_mat[last_idx, :].T) > 0).sum()*0.5)/res.shape[0]
    else:
        pred = np.nan

    lbl = g_i.get_label(last_idx)
    if lbl > 0:
        pred /= lbl
        lbl = 1
    return pred, lbl


for b in bs:
    for phi in phis:
        all_test = map(lambda i: (g.insert(order_no_test[i], h_test[i, :],
                                           bsk_label_test.iloc[i, 0],
                                           r_test[i, :], False),
                                  b, phi),
                       np.arange(0, min(max_num_test, len(order_no_test))))

        if parallel == 0:
            pred_rst = list()
            for g_i in all_test:
                pred_rst.append(eval_i(g_i))
        elif parallel == 1:
            pool = Pool(processes=n_cpu)
            pred_rst = \
                    pool.imap_unordered(eval_i, 
                                        ((g.insert(order_no_test[i], h_test[i, :], bsk_label_test.iloc[i, 0],
                                                   r_test[i,:], False),
                                          b, phi)
                                         for i in np.arange(1000, 1000+min(max_num_test, len(order_no_test)))),
                                        chunksize=chunk_size)
            pred_rst = list(pred_rst)
            pool.close()
        elif parallel == 2:
            pred_rst = list(futures.map(eval_i, all_test))

        pred_rst = pd.DataFrame(pred_rst, columns=['pred_prob', 'obs'])
        pred_rst.to_csv("rst/prediction.csv")

        fpr, tpr, thr_fpr_tpr = metrics.roc_curve(pred_rst['obs'], pred_rst['pred_prob'], pos_label=True)
        roc_auc = metrics.auc(fpr, tpr)

        prec, recall, thr_prec_recall = \
                metrics.precision_recall_curve(pred_rst['obs'], pred_rst['pred_prob'], pos_label=True)
        f1 = (1+0.25)*prec*recall/(0.25*prec + recall)
        f1[np.isnan(f1)] = 0
        max_f1_idx = np.argmax(f1)
        print("%f, %f: %f, %f, %f, %f" %
              (b, phi, roc_auc, prec[max_f1_idx], recall[max_f1_idx], f1[max_f1_idx]), thr_prec_recall[max_f1_idx])
        with open('rst/b_phi_tune.csv', 'a+') as f:
            f.write("%f, %f: %f, %f, %f, %f\n" %
                    (b, phi, roc_auc, prec[max_f1_idx], recall[max_f1_idx], f1[max_f1_idx]))
#
