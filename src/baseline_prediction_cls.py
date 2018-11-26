import pandas as pd
import pickle
import numpy as np
from sklearn import metrics
import matplotlib.pylab as plt
from src.functions import f_point_5


class BaseLinePredictor:
    """
    A baseline prediction class
    """
    def __init__(self, method='Jaccard', type='Unnormalized'):
        """
        The constructor of the baseline predictor
        :param method: only Jaccard method is supported
        :param type: one of the two options ('Unnormalized', 'Normalized')
        """
        self.method = method
        self.type = type
        self.h_train = None
        self.bsk_label_train = None
        self.__union_train = None
        self.multi_item_bsk = None
        self.ratio = None

    def fit(self, h_train, bsk_label_train, multi_item_bsk = None, ratio = None):
        """
        The train method of class
        :param h_train: a csr_matrix
        :param bsk_label_train: a single column data frame
        :return: None
        """
        h_train = h_train.sign()
        self.h_train = h_train
        self.bsk_label_train = bsk_label_train.values
        self.__union_train = h_train.sum(1)

        if self.type == "Normalized":
            assert multi_item_bsk is not None, "multi_item_bsk must be supplied for normalization!"

            self.multi_item_bsk = multi_item_bsk
            multi_idx = np.in1d(bsk_label_train.index.values, multi_item_bsk.index.values)
            if ratio is None:
                self.ratio = bsk_label_train[multi_idx].sum()[0]/multi_idx.sum() /\
                    (bsk_label_train[np.logical_not(multi_idx)].sum()[0]/np.logical_not(multi_idx).sum())
            else:
                self.ratio = ratio
            self.bsk_label_train = self.bsk_label_train.astype(np.float)
            self.bsk_label_train[multi_idx, :] = self.bsk_label_train[multi_idx, :] / self.ratio
        else:
            self.ratio = 1

    def predict(self, h_test, test_bsk_name=None):
        """
        Predict the result based on given h_test
        :param h_test: a csr_matrix
        :return: a list of prediction (continuous score)
        """
        h_test = h_test.sign()
        wgt = np.ones((h_test.shape[0]))
        if self.type == "Normalized":
            assert test_bsk_name is not None, "test_bsk_name must be supplied for normalization"
            multi_idx = np.in1d(test_bsk_name, self.multi_item_bsk.index.values)

            wgt[multi_idx] = self.ratio

        intersect = self.h_train * h_test.T
        union2 = h_test.sum(1)

        pred = []
        for i in range(len(union2)):
            ja = intersect[:, i] / (self.__union_train + union2[i] - intersect[:, i])  # Jaccard Index
            pred.append((ja.T * self.bsk_label_train/ja.sum())[0, 0] * wgt[i])
        return pred

    def pred_test_based_on_valid(self, h_validate, bsk_label_valid, h_test, bsk_label_test):
        """

        :param h_validate:
        :param bsk_label_valid:
        :param h_test:
        :return: optimal prec, recall, f_0.5, auc, fpr, tpr
        """
        pred_val = self.predict(h_validate, bsk_label_valid.index.values)

        prec, rec, thr = metrics.precision_recall_curve(bsk_label_valid, pred_val)
        f = f_point_5(prec, rec)
        thr_opt = thr[np.argmax(f)]

        pred_test = self.predict(h_test, bsk_label_test.index.values)
        prec, rec, thr = metrics.precision_recall_curve(bsk_label_test, pred_test)
        f = f_point_5(prec, rec)
        idx = np.where(thr <= thr_opt)[0][-1]

        fpr, tpr, _ = metrics.roc_curve(bsk_label_test, pred_test)
        auc = metrics.auc(fpr, tpr)

        return prec[idx], rec[idx], f[idx], auc, fpr, tpr, thr[idx]


if __name__ == "__main__":

    # load data
    with open("data/clean/h_train.pkl", 'rb') as f:
        h_train = pickle.load(f)
    with open("data/clean/h_validate.pkl", 'rb') as f:
        h_validate = pickle.load(f)
    with open("data/clean/h_test.pkl", 'rb') as f:
        h_test = pickle.load(f)

    bsk_label_train = pd.read_pickle("data/clean/bsk_label_train.pkl")
    bsk_label_valid = pd.read_pickle("data/clean/bsk_label_validate.pkl")
    bsk_label_test = pd.read_pickle("data/clean/bsk_label_test.pkl")

    # unnormalized
    p = BaseLinePredictor()
    p.fit(h_train, bsk_label_train)

    prec, rec, f, auc, fpr, tpr, thr = \
        p.pred_test_based_on_valid(h_validate, bsk_label_valid, h_test, bsk_label_test)

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'b--')

    print("auc, prec, rec, f1, thr:")
    print("%f, %f, %f, %f, %f" % (auc, prec, rec, f, thr))

    # normalized
    multi_item_bsk = pd.read_pickle("data/clean/multi_item_bsk_return_label.pkl")
    p_1 = BaseLinePredictor(type="Normalized")
    p_1.fit(h_train, bsk_label_train, multi_item_bsk, 2)

    prec_1, rec_1, f_1, auc_1, fpr_1, tpr_1, thr =\
        p_1.pred_test_based_on_valid(h_validate, bsk_label_valid, h_test, bsk_label_test)

    plt.plot(fpr_1, tpr_1)
    plt.plot([0, 1], [0, 1], 'b--')

    print("auc, prec, rec, f1, thr:")
    print("%f, %f, %f, %f, %f" % (auc_1, prec_1, rec_1, f_1, thr))

