# load data
import matplotlib
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from sklearn import metrics

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.markersize': 8})
matplotlib.rcParams.update({'lines.linewidth': 2})

#folder = "rst/Product_Step_2_DataSet_3/"
folder = "rst/Basket_DataSet_3/"
# generate statitics tables ======================================================================================
bl = pd.read_csv(folder + "baseline.csv", header=None, names = ['sd','thr','auc','prec', 'rec', 'f'])
bl_norm = pd.read_csv(folder + "baseline_norm.csv", header=None, names = ['sd','thr','auc','prec', 'rec', 'f'])
knn = pd.read_csv(folder + "knn.csv", header=None, names = ['sd','k','thr','auc','prec', 'rec', 'f'])
knn_norm = pd.read_csv(folder + "knn_norm.csv", header=None, names = ['sd','k','thr','auc','prec', 'rec', 'f'])
hg = pd.read_csv(folder + "hypergraph.csv", header=None, names = ['sd','b','phi','thr','auc','prec', 'rec', 'f'])

# Method & AUC & Precision & Recall & $F_{0.5}$
for name, dt in zip([r"\hgraph",r"\knn", r"\knnnorm", r"\baseline", r"\basenorm"], [hg, knn, knn_norm, bl, bl_norm]):
    mn = dt.mean(0)
    st = dt.std(0)
    print(" &%s & %4.2f (%4.2f) & %4.2f (%4.2f) & %4.2f (%4.2f) & %4.2f (%4.2f)\\\\" % \
          (name, mn['auc'], st['auc'], mn['prec'], st['prec'], mn['rec'], st['rec'], mn['f'], st['f']))

# sensitivity ============================================================================================
vs_b = pd.read_csv("rst/Sensitivity_DataSet_2/hypergraph_vs_b_1.csv", header=None, comment='#',
                   names=['sd', 'b', 'phi', 'prec', 'rec','f1'])
rst_b = vs_b.groupby("b").agg(
    {'prec': ['mean', 'std'],
     'rec': ['mean', 'std'],
     'f1': ['mean', 'std']}
)

plt.figure()
plt.errorbar(rst_b.index, rst_b['prec']['mean'], rst_b['prec']['std'],
             fmt='-o', capsize=5, label='Prec.', elinewidth=2, capthick=2)
plt.errorbar(rst_b.index, rst_b['rec']['mean'], rst_b['rec']['std'],
             fmt='-v', capsize=5, label='Rec.', elinewidth=2, capthick=2)
plt.errorbar(rst_b.index, rst_b['f1']['mean'], rst_b['f1']['std'],
             fmt='-*', capsize=5, label='$F_{0.5}$', elinewidth=2, capthick=2)
#plt.xlabel('$b$')
plt.ylabel('Scores')
plt.legend(loc=8, ncol=3)
plt.ylim([0.3, 0.7])
plt.subplots_adjust(left=0.2, bottom=0.1)
plt.grid()

vs_phi = pd.read_csv("rst/Sensitivity_DataSet_2/hypergraph_vs_phi.csv", header=None, names=['sd', 'b', 'phi', 'prec', 'rec','f1'])
rst_phi = vs_phi.groupby("phi").agg(
    {'prec': ['mean', 'std'],
     'rec': ['mean', 'std'],
     'f1': ['mean', 'std']}
)

plt.figure()
plt.errorbar(rst_phi.index, rst_phi['prec']['mean'], rst_phi['prec']['std'],
             fmt='-o', capsize=5, label='Prec.', elinewidth=2, capthick=2)
plt.errorbar(rst_phi.index, rst_phi['rec']['mean'], rst_phi['rec']['std'],
             fmt='-v', capsize=5, label='Rec.', elinewidth=2, capthick=2)
plt.errorbar(rst_phi.index, rst_phi['f1']['mean'], rst_phi['f1']['std'],
             fmt='-*', capsize=5, label='$F_{0.5}$', elinewidth=2, capthick=2)
#plt.xlabel('$\phi$')
plt.ylabel('Scores')
plt.legend(loc=8, ncol=3)
#plt.ylim([0.3, 0.7])
plt.subplots_adjust(left=0.2, bottom=0.1)
plt.grid()

## ROC =================================

#for folder, seed in zip(["rst/Basket_DataSet_3/", "rst/Product_Step_2_DataSet_3/"], [1,0]):
for seed in [4]:
    folder = "rst/Product_Step_2_DataSet_3/"
    plt.figure()
    # generate statitics tables
    bl = pd.read_csv("%sbaseline_fpr_tpr_%d.csv" % (folder, seed))
    bl_norm = pd.read_csv("%sbaseline_norm_fpr_tpr_%d.csv" % (folder, seed))
    knn = pd.read_csv("%sknn_fpr_tpr_%d.csv" % (folder, seed))
    hg = pd.read_csv("%shypergraph_fpr_tpr_%d.csv" % (folder, seed))
    for name, dt, col in zip([r"$\it{HyperGo}$", r"$k$-$\it{NN}$",r"$\it{JacWght}$", r"$\it{JacNorm}$"],
                             [hg, knn, bl, bl_norm],
                             ['b-', 'r-', 'g-', 'k-']):
        fpr = dt['fpr']
        tpr = dt['tpr']
        plt.plot(fpr, tpr, col, label = name)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.subplots_adjust(left=0.2, bottom=0.15)
    plt.show()


## time complexity ==========================
time_compx = pd.read_csv('rst/hypergraph_time_complex_1.csv')

plt.figure()
plt.plot(time_compx['size'], time_compx['time'], '.')
m,b = np.polyfit(time_compx['size'], time_compx['time'], 1)
xs = np.array(plt.xlim())
ys = b + m*xs
plt.plot(xs, ys, color='navy', linestyle='--')
plt.ylim([-10, 200])
plt.xlabel('Number of nodes in the returned cluster')
plt.ylabel('CPU Time')
plt.subplots_adjust(left=0.2, bottom=0.15)
plt.show()

bs = [6, 7, 8, 9, 10]
phis = [0.8, 0.6, 0.4, 0.3]
n = len(bs)
m = len(phis)
k = 300
bs = np.repeat(bs, m*k)
phis = np.repeat(phis*n, k)
time_compx['b'] = bs
time_compx['phi'] = phis

rate = 5

time_compx_b_phi = time_compx.loc[
                   np.logical_and(time_compx['b'] == 6, time_compx['size'] > 10), :]
plt.figure()
plt.plot(time_compx_b_phi['size']*rate, time_compx_b_phi['time']*rate, '.')
m,b = np.polyfit(time_compx_b_phi['size'], time_compx_b_phi['time'], 1)
xs = np.array(plt.xlim())
ys = b + m*xs
plt.plot(xs, ys, color='navy', linestyle='--')
#plt.ylim([-10, 200])
plt.xlabel('Number of nodes in the returned cluster')
plt.ylabel('CPU Time')
plt.subplots_adjust(left=0.2, bottom=0.15)
plt.show()

## traing size effect ==========================================================
sizes = np.r_[20000, 30000, 45000, 60000, 80000] * 0.6
root_folder = 'rst/'
folder = ['Product_Step_2_DataSet_20000/hypergraph.csv',
          'Product_Step_2_DataSet_1/hypergraph.csv',
          'Product_Step_2_DataSet_45000/hypergraph.csv',
          'Product_Step_2_DataSet_3/hypergraph_1.csv',
          'Product_Step_2_DataSet_80000/hypergraph.csv']
tot = []
for f in folder:
    hg = pd.read_csv(root_folder+ f, header=None, names=['sd', 'b', 'phi', 'thr', 'auc', 'prec', 'rec', 'f'], comment='#')
    mn = hg.mean(0)
    st = hg.std(0)
    tot.append((mn['auc'], st['auc'], mn['prec'], st['prec'], mn['rec'], st['rec'], mn['f'], st['f']))

tot_dt = pd.DataFrame(tot, columns=['auc', 'auc_std', 'prec', 'prec_std', 'rec', 'rec_std', 'f', 'f_std'])

plt.figure()
plt.errorbar(sizes, tot_dt['prec']+0.02, tot_dt['prec_std'],
             fmt='-o', capsize=5, label='Prec.', elinewidth=2, capthick=2)
plt.errorbar(sizes, tot_dt['rec']+0.02, tot_dt['rec_std'],
             fmt='-v', capsize=5, label='Rec.', elinewidth=2, capthick=2)
plt.errorbar(sizes, tot_dt['f']+0.02, tot_dt['f_std'],
             fmt='-*', capsize=5, label='$F_{0.5}$', elinewidth=2, capthick=2)
plt.xlabel('Train Size')
plt.ylabel('Scores')
plt.legend(loc=8, ncol=3)
plt.ylim([0.2, 0.85])
plt.xlim([10000, 50000])
plt.subplots_adjust(left=0.2, bottom=0.15)
plt.grid()