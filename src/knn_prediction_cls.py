import pandas as pd
import pickle
import numpy as np
from sklearn import metrics, neighbors
# import matplotlib.pylab as plt
from src.functions import f_point_5, argmax


class knn_Predictor:
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
        self.bsk_label_train = None
        self.clf = None

    def fit(self, h_train, bsk_label_train):
        """
        The train method of class
        :param h_train: a csr_matrix
        :param bsk_label_train: a single column data frame
        :return: None
        """
        assert self.k is not None, "k cannot be none before train"
        self.h_train = h_train.sign()
        if isinstance(bsk_label_train, pd.DataFrame):
            bsk_label_train = bsk_label_train.values
        self.bsk_label_train = bsk_label_train
        self.clf = neighbors.KNeighborsClassifier(self.k, weights='uniform')
        self.clf.fit(h_train, np.squeeze(self.bsk_label_train))

    def predict(self, h_test):
        """
        Predict the result based on given h_test
        :param h_test: a csr_matrix
        :return: a list of prediction (continuous score)
        """
        assert self.clf is not None, "The model need to be trained before used for prediction"

        h_test = h_test.sign()
        pred = self.clf.predict(h_test)
        return pred

    def pred_test_based_on_valid(self, h_validate, bsk_label_valid, h_test, bsk_label_test, ks):
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
            self.fit(self.h_train, self.bsk_label_train)
            pred_val = self.predict(h_validate)

            prec, rec, thr = metrics.precision_recall_curve(bsk_label_valid, pred_val)
            f = f_point_5(prec, rec)
            fpr, tpr, _ = metrics.roc_curve(bsk_label_valid, pred_val)
            auc = metrics.auc(fpr, tpr)

            thr_opt.append(thr[np.argmax(f)])
            best_f.append(np.max(f))
            best_auc.append(auc)

        idx_k = argmax(best_f, best_auc)[0]
        self.k = ks[idx_k]
        self.fit(self.h_train, self.bsk_label_train)
        pred_test = self.predict(h_test)

        prec, rec, thr = metrics.precision_recall_curve(bsk_label_test, pred_test)
        f = f_point_5(prec, rec)
        idx_f = np.where(thr <= thr_opt[idx_k])[0][-1]

        fpr, tpr, thr = metrics.roc_curve(bsk_label_test, pred_test)
        auc = metrics.auc(fpr, tpr)

        return prec[idx_f], rec[idx_f], f[idx_f], auc, fpr, tpr, thr[idx_f], ks[idx_k]

if __name__ == "__main__":
    import matplotlib.pylab as plt
    with open("../data/h_train.pkl", 'rb') as f:
        h_train = pickle.load(f)
    with open("../data/h_validate.pkl", 'rb') as f:
        h_validate = pickle.load(f)
    with open("../data/h_test.pkl", 'rb') as f:
        h_test = pickle.load(f)
    bsk_label_train = pd.read_pickle('../data/bsk_label_train.pkl')
    bsk_label_valid = pd.read_pickle("../data/bsk_label_validate.pkl")
    bsk_label_test = pd.read_pickle('../data/bsk_label_test.pkl')

    # k-d tree
    p = knn_Predictor(k=5)
    p.fit(h_train, bsk_label_train)
    prec, rec, f, auc, fpr, tpr, thr, k = \
        p.pred_test_based_on_valid(h_validate, bsk_label_valid, h_test, bsk_label_test,
                                   [1, 3, 5, 10, 15, 20, 25])

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'b--')

    print("auc, prec, rec, f1, thr, k:")
    print("%f, %f, %f, %f, %f, %f" % (auc, prec, rec, f, thr, k))

