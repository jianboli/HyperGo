import pandas as pd
from sklearn import metrics
import matplotlib.pylab as plt


def plot_roc(fpr, tpr, roc_auc):
    """Generate the ROC Curve
    
    Arguments:
        fpr {numpy 1 d array} -- [false positive rate]
        tpr {numpy 1d array} -- [ture postive rate]
        roc_auc {float} -- [Area under roc curve]
    """

    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

def plot_prec_recall(prec, recall, maximum_f1):
    """Generate the precision recall curve
    
    Arguments:
        prec {[numpy 1 d array]} -- [precision]
        recall {[numpy 1 d array]} -- [recall]
        maximum_f1 {[float]} -- [f1 score]
    """

    lw = 2
    plt.plot(recall, prec, color='darkorange',
             lw=lw, label='maximum F1 %0.2f ' % maximum_f1)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")


#plt.figure()
prd = pd.read_csv('rst/prediction.csv')
fpr, tpr, thr_fpr_tpr = metrics.roc_curve(prd['obs'], prd['pred_prob'], pos_label=True)
roc_auc = metrics.auc(fpr, tpr)
#plot_roc(fpr, tpr, roc_auc)
print(roc_auc)

#plt.figure()
prec, recall, thr_prec_recall = metrics.precision_recall_curve(prd['obs'], prd['pred_prob'], pos_label=True)
f1 = 2*prec*recall/(prec + recall)
#plot_prec_recall(prec, recall, max(f1))
print(max(f1))
#plt.show()
