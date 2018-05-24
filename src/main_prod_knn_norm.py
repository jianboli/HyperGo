import pickle
import numpy as np
import pandas as pd
# from scoop import futures

from src.pre_process_cls import SplitTrainValidateTest
from src.baseline_prediction_2_step_cls import BaseLinePredictor
from src.knn_prediction_2_step_norm_cls import knn_Predictor_Norm
#from src.hg_prediction_cls import HypergraphPredictor
from src.hg_prediction_2_step_cls import HypergraphPredictor
from itertools import compress

input_path = 'data/clean/'
output_path = 'data/split_'
result_path = 'rst'
seeds = range(10)
step = 2 # step 1, conditional probability
         # step 2, marginal probability
         # for basket level prediction use 2

split = SplitTrainValidateTest(0.6, 0.2)
split.read_data(input_path)

for sd in seeds:
    print("Start of seed %d" % sd)
    # random split =======================================
    output_path_sd = output_path+str(sd)+str("/")
    split.random_split(sd)
    split.export_data(output_path_sd)

    # Load data ==========================================
    with open(output_path_sd + "order_no_train.pkl", 'rb') as f:
        order_no_train = pickle.load(f)
    with open(output_path_sd + "khk_ean_train.pkl", 'rb') as f:
        khk_ean_train = pickle.load(f)
    bsk_label_train = pd.read_pickle(output_path_sd + "bsk_label_train.pkl")
    return_rate_train = pd.read_pickle(output_path_sd + "return_rate_train.pkl")
    with open(output_path_sd + "h_train.pkl", 'rb') as f:
        h_train = pickle.load(f)
    with open(output_path_sd + "r_train.pkl", 'rb') as f:
        r_train = pickle.load(f)

    with open(output_path_sd + "h_validate.pkl", 'rb') as f:
        h_validate = pickle.load(f)
    with open(output_path_sd + "r_validate.pkl", 'rb') as f:
        r_validate = pickle.load(f)
    with open(output_path_sd + "order_no_validate.pkl", 'rb') as f:
        order_no_validate = pickle.load(f)
    bsk_label_validate = pd.read_pickle(output_path_sd + "bsk_label_validate.pkl")


    with open(output_path_sd + "h_test.pkl", 'rb') as f:
        h_test = pickle.load(f)
    with open(output_path_sd + "r_test.pkl", 'rb') as f:
        r_test = pickle.load(f)
    with open(output_path_sd + "order_no_test.pkl", 'rb') as f:
        order_no_test = pickle.load(f)
    bsk_label_test = pd.read_pickle(output_path_sd + "bsk_label_test.pkl")

    multi_item_bsk = pd.read_pickle(input_path + "multi_item_bsk_return_label.pkl")
    bsk_ret_item_collection = pd.read_pickle(input_path + "bsk_return_item_collection.pkl")

    # Ground Truth ============================================================
    if step == 1:
        idx = np.where(bsk_label_validate['RET_Items'].values)[0]
        h_validate = h_validate[idx,:]
        order_no_validate = [order_no_validate[i] for i in idx]
        r_validate = r_validate[idx, :]
        bsk_label_validate = bsk_label_validate.iloc[idx,:]

        idx = np.where(bsk_label_test['RET_Items'].values)[0]
        h_test = h_test[idx,:]
        order_no_test = [order_no_test[i] for i in idx]
        r_test = r_test[idx, :]
        bsk_label_test = bsk_label_test.iloc[idx,:]


    # k-nn ================================================
    multi_item_bsk = multi_item_bsk.index.values
    multi_idx_train = np.in1d(bsk_label_train.index.values, multi_item_bsk)
    p = knn_Predictor_Norm(k=5)
    p.fit(h_train, bsk_label_train, r_train, multi_idx_train, step=step)

    multi_idx_validate = np.in1d(bsk_label_validate.index.values, multi_item_bsk)
    multi_idx_test = np.in1d(bsk_label_test.index.values, multi_item_bsk)
    prec, rec, f, auc, fpr, tpr, thr, k = \
        p.pred_test_based_on_valid_prod(h_validate, bsk_label_validate, r_validate, multi_idx_validate,
                                        h_test, bsk_label_test, r_test, multi_idx_test,
                                        [10, 15, 20, 25])
    with open(result_path + "/knn.csv", 'a+') as ff:
        ff.write("%f, %f, %f, %f, %f, %f, %f\n" % (sd, k, thr, auc, prec, rec, f))
    fpr_tpr = pd.DataFrame({'fpr':fpr,  'tpr':tpr})
    fpr_tpr.to_csv("%s/knn_fpr_tpr_%d.csv" % (result_path, sd))
