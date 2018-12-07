import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
# from scoop import futures

from src.pre_process_cls import SplitTrainValidateTest
from src.functions import f_point_5
from src.hg_prediction_2_step_cls import HypergraphPredictor
from itertools import compress
if __name__ == "__main__":
    input_path = '../data/'
    output_path = '../data/split_'
    result_path = '../rst'

    opt_result_path = 'rst/Basket_DataSet_1/' # an optimal values is supposed to have been supplied

    seeds = range(2)
    step = 2 # step 1, conditional probability
             # step 2, marginal probability
             # for basket level prediction use 2

    split = SplitTrainValidateTest(0.6, 0.2)
    split.read_data(input_path)


    for sd in seeds:
        vs_b = []
        vs_phi = []
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

        # multi_item_bsk = pd.read_pickle(input_path + "multi_item_bsk_return_label.pkl")

        # bsk_ret_item_collection = pd.read_pickle(input_path + "bsk_return_item_collection.pkl")

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

        # hyper graph ====================================
        p = HypergraphPredictor(max_num_test=300, parallel="Multi", n_cpu=7, chunk_size=1)
        p.fit(h_train, bsk_label_train, order_no_train, khk_ean_train,
              return_rate_train, r_train, ratio=None, step=step)
        bs = [4.5, 5, 6]
        phis = []
        best_b = 5 # opt_hg.loc[idx, 'b']
        best_phi = 0.9 # opt_hg.loc[idx, 'phi']
        best_thr = 0.62
        n = len(bs)
        m = len(phis)

        for i in range(n):
            b = bs[i]
            phi = best_phi
            print(b, phi)
            pred_rst = p.predict(h_test, bsk_label_test, order_no_test, r_test, b, phi)
            y = pred_rst['obs'].values
            y_hat = (pred_rst['pred_prob'].values > best_thr)
            tp = (np.logical_and(y, y_hat)).sum()
            prec = tp/sum(y_hat)
            rec = tp/sum(y)
            f1 = f_point_5(prec, rec)
            vs_b.append((sd, b, phi, prec, rec, f1))

        for j in range(m):
            b = best_b
            phi = phis[j]
            print(b, phi)
            pred_rst = p.predict(h_test, bsk_label_test, order_no_test, r_test, b, phi)
            y = pred_rst['obs'].values
            y_hat = (pred_rst['pred_prob'].values > best_thr)
            tp = (np.logical_and(y, y_hat)).sum()
            prec = tp/sum(y_hat)
            rec = tp/sum(y)
            f1 = f_point_5(prec, rec)
            vs_phi.append((sd, b, phi, prec, rec, f1))

        with open(result_path + "/hypergraph_vs_b.csv", 'a+') as ff:
            for sd, b, phi, prec, rec, f in vs_b:
                ff.write("%f, %f, %f, %f, %f, %f\n" % (sd, b, phi, prec, rec, f))

        with open(result_path + "/hypergraph_vs_phi.csv", 'a+') as ff:
            for sd, b, phi, prec, rec, f in vs_phi:
                ff.write("%f, %f, %f, %f, %f, %f\n" % (sd, b, phi, prec, rec, f))
