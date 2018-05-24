import pandas as pd
import pickle
import numpy as np
from sklearn import metrics, neighbors
# import matplotlib.pylab as plt
from src.functions import f_point_5, argmax


class knn_Predictor_Norm:
    """
    A knn prediction class
    """
    def __init__(self, k=None):
        """
        The constructor of the baseline predictor
        :param method: only Jaccard method is supported
        :param type: one of the two options ('Unnormalized', 'Normalized')
        """
        self.k = k
        self.h_train = None
        self.r_train = None
        self.bsk_label_train = None
        self.clf = None
        self.step = None
        self.multi_labels = None
        self.ratio = None

    def fit(self, h_train, bsk_label_train, r_train, multi_item_bsk, ratio=None, step=2):
        """
        The train method of class
        :param h_train: a csr_matrix
        :param bsk_label_train: a single column data frame
        :return: None
        """
        assert self.k is not None, "k cannot be none before train"
        self.h_train = h_train.sign()
        self.r_train = r_train
        if isinstance(bsk_label_train, pd.DataFrame):
            bsk_label_train = bsk_label_train.values
        self.bsk_label_train = np.squeeze(bsk_label_train)
        self.clf = neighbors.KNeighborsClassifier(self.k, weights='uniform')
        self.clf.fit(h_train, np.squeeze(self.bsk_label_train))
        self.step = step
        self.multi_labels_train = multi_item_bsk
        if ratio is None:
            ratio = bsk_label_train[multi_item_bsk].sum()/multi_item_bsk.sum() / \
                    (bsk_label_train[np.logical_not(multi_item_bsk)].sum()/np.logical_not(multi_item_bsk).sum())
        self.ratio = ratio


    def predict(self, h_test, r_test, mulit_labels_test):
        """
        Predict the result based on given h_test
        :param h_test: a csr_matrix
        :return: a list of prediction (continuous score)
        """
        assert self.clf is not None, "The model need to be trained before used for prediction"
        h_test = h_test.sign()

        h_mat = self.h_train.sign()
        r_mat = self.r_train.sign()
        tot_ret_rate = (r_mat.sum(0)/h_mat.sum(0)).A1
        pred = []

        ratio = self.ratio
        for i in range(h_test.shape[0]):
            nn = self.clf.kneighbors(h_test[i, :], self.k)[1][0]
            if self.step == 1:
                pred_bsk = 1
            else:
                res_label = 1-self.bsk_label_train[nn]
                res_multi = self.multi_labels_train[nn]

                a = res_label.dot(1-res_multi)/len(nn)
                c = res_label.dot(res_multi)/len(nn)
                pred_i = ((1-a)*ratio + (1-c) - np.sqrt((1-a)**2*ratio**2+(1-c)**2+2*(a*c+(a+c)-1)*ratio))/(2*ratio)

                if mulit_labels_test[i]:
                    pred_i = pred_i * ratio

            res_h = self.h_train[nn, :].sign()
            res_r = self.r_train[nn, :].sign()
            with np.errstate(divide='ignore',invalid='ignore'):
                pred_prod_i = (res_r.T.dot(1-res_label))/(res_h.T.dot(1-res_label))
            idx = np.isnan(pred_prod_i)
            pred_prod_i[idx] = tot_ret_rate[idx]
            res_h1 = (h_test[i, :] > 1).todense().A1+1
            pred_prod_i = pred_prod_i * res_h1
            idx = (h_test[i, :].todense().A1 > 0)
            pred_prod_i = pred_prod_i[idx] * pred_i

            pred.append((pred_i, r_test[i, idx].sum() > 0,
                         pred_prod_i, r_test[i, idx].todense().A1 > 0))
        pred_rst = pd.DataFrame(pred, columns=['pred_prob', 'obs', 'pred_prob_prod', 'obs_prod'])
        return pred_rst

    def pred_test_based_on_valid(self, h_validate, bsk_label_valid, r_validate, multi_labels_validate,
                                      h_test, bsk_label_test, r_test, multi_labels_test, ks):
        """
        Loop through a ks is given, otherwise, k could be specified before calling this method
        :param h_validate:
        :param bsk_label_valid:
        :param h_test:
        :return: optimal prec, recall, f_0.5, auc, fpr, tpr
        """

        best_f = []
        best_auc = []
        thr_opt = []
        for k in ks:
            self.k = k
            self.fit(self.h_train, self.bsk_label_train, self.r_train,
                     self.multi_labels_train, step=self.step)
            pred_val = self.predict(h_validate, r_validate, multi_labels_validate)

            obs_prod = pred_val['obs']
            pred_prod = pred_val['pred_prob']
            prec, rec, thr = metrics.precision_recall_curve(obs_prod, pred_prod, pos_label=True)

            f = f_point_5(prec, rec)
            thr_opt.append(thr[np.argmax(f)])
            best_f.append(np.max(f))
            fpr, tpr, _ = metrics.roc_curve(obs_prod, pred_prod, pos_label=True)
            auc = metrics.auc(fpr, tpr)
            best_auc.append(auc)

        idx_k = argmax(best_f, best_auc)[0]
        self.k = ks[idx_k]
        self.fit(self.h_train, self.bsk_label_train, self.r_train,
                 self.multi_labels_train, self.step)
        pred_test = self.predict(h_test, r_test, multi_labels_test)

        obs_prod = pred_test['obs']
        pred_prod = pred_test['pred_prob']
        prec, rec, thr = metrics.precision_recall_curve(obs_prod, pred_prod, pos_label=True)
        f = f_point_5(prec, rec)
        idx_f = np.where(thr <= thr_opt[idx_k])[0][-1]

        fpr, tpr, _ = metrics.roc_curve(obs_prod, pred_prod)
        auc = metrics.auc(fpr, tpr)

        return prec[idx_f], rec[idx_f], f[idx_f], auc, fpr, tpr, thr[idx_f], ks[idx_k]



    def pred_test_based_on_valid_prod(self, h_validate, bsk_label_valid, r_validate, multi_labels_validate,
                                 h_test, bsk_label_test, r_test, multi_labels_test, ks):
        """
        Loop through a ks is given, otherwise, k could be specified before calling this method
        :param h_validate:
        :param bsk_label_valid:
        :param h_test:
        :return: optimal prec, recall, f_0.5, auc, fpr, tpr
        """

        best_f = []
        best_auc = []
        thr_opt = []
        for k in ks:
            self.k = k
            self.fit(self.h_train, self.bsk_label_train, self.r_train,
                     self.multi_labels_train, step=self.step)
            pred_val = self.predict(h_validate, r_validate, multi_labels_validate)

            obs_prod = np.array([item for l in pred_val['obs_prod'] for item in l])
            pred_prod = np.array([item for l in pred_val['pred_prob_prod'] for item in l])
            prec, rec, thr = metrics.precision_recall_curve(obs_prod, pred_prod, pos_label=True)

            f = f_point_5(prec, rec)
            thr_opt.append(thr[np.argmax(f)])
            best_f.append(np.max(f))
            fpr, tpr, _ = metrics.roc_curve(obs_prod, pred_prod, pos_label=True)
            auc = metrics.auc(fpr, tpr)
            best_auc.append(auc)

        idx_k = argmax(best_f, best_auc)[0]
        self.k = ks[idx_k]
        self.fit(self.h_train, self.bsk_label_train, self.r_train,
                 self.multi_labels_train, self.step)
        pred_test = self.predict(h_test, r_test, multi_labels_test)

        obs_prod = np.array([item for l in pred_test['obs_prod'] for item in l])
        pred_prod = np.array([item for l in pred_test['pred_prob_prod'] for item in l])
        prec, rec, thr = metrics.precision_recall_curve(obs_prod, pred_prod, pos_label=True)
        f = f_point_5(prec, rec)
        idx_f = np.where(thr <= thr_opt[idx_k])[0][-1]

        fpr, tpr, _ = metrics.roc_curve(obs_prod, pred_prod)
        auc = metrics.auc(fpr, tpr)

        return prec[idx_f], rec[idx_f], f[idx_f], auc, fpr, tpr, thr[idx_f], ks[idx_k]


if __name__ == "__main__":

    with open("data/clean/h_train.pkl", 'rb') as f:
        h_train = pickle.load(f)
    with open("data/clean/r_train.pkl", 'rb') as f:
        r_train = pickle.load(f)
    with open("data/clean/h_validate.pkl", 'rb') as f:
        h_validate = pickle.load(f)
    with open("data/clean/h_test.pkl", 'rb') as f:
        h_test = pickle.load(f)
    bsk_label_train = pd.read_pickle('data/clean/bsk_label_train.pkl')
    bsk_label_valid = pd.read_pickle("data/clean/bsk_label_validate.pkl")
    bsk_label_test = pd.read_pickle('data/clean/bsk_label_test.pkl')

    with open("data/clean/r_validate.pkl", 'rb') as f:
        r_validate = pickle.load(f)
    with open("data/clean/r_test.pkl", 'rb') as f:
        r_test = pickle.load(f)
    # k-d tree
    p = knn_Predictor_Norm(k=5)
    multi_item_bsk = pd.read_pickle("data/clean/multi_item_bsk_return_label.pkl")
    multi_item_bsk = multi_item_bsk.index.values
    multi_idx_train = np.in1d(bsk_label_train.index.values, multi_item_bsk)
    p.fit(h_train, bsk_label_train, r_train, multi_idx_train, step=2)
    multi_idx_valid = np.in1d(bsk_label_valid.index.values, multi_item_bsk)
    multi_idx_test = np.in1d(bsk_label_test.index.values, multi_item_bsk)

    prec, rec, f, auc, fpr, tpr, thr, k = \
        p.pred_test_based_on_valid(h_validate, bsk_label_valid, r_validate, multi_idx_valid,
                                        h_test, bsk_label_test, r_test, multi_idx_test,
                                        [3, 5, 10, 15, 20, 25])

    #  plt.plot(fpr, tpr)
    # plt.plot([0, 1], [0, 1], 'b--')

    print("auc, prec, rec, f1, thr, k:")
    print("%f, %f, %f, %f, %f, %f" % (auc, prec, rec, f, thr, k))

