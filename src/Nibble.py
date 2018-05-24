#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:41:42 2017

@author: bob
"""

import bisect
# import matplotlib.pylab as plt
import pandas as pd
from scipy import sparse
# %% define Nibble ===========================================================
import numpy as np

from src.hyper_graph import HyperGraph

def Truncat(p, threshold):
    """
        define the truncation function in step 2
    """
    for idx in p.nonzero()[0]:
        if (p[idx, 0] <= threshold[idx]):
            p[idx, 0] = 0
    return p

def ApproxSJ2(g, q):
    idxJ = q.nonzero()
    qNNZ = q[idxJ]
    # TODO: We added divided by Pi here
    pi = g.stationary_dist[idxJ[0]]
    if isinstance(qNNZ, sparse.csr.csr_matrix):
        qNNZ = (qNNZ/pi).todense()

    qNNZ = np.squeeze(np.asarray(qNNZ))
    idx = np.argsort(qNNZ)[::-1]
    return idxJ[0][idx]


def Nibble(g:HyperGraph, v, b, phi, debug = False):
    """
       Function Nibble is the implementation of the adaptive Nibble algorithm
       g: the graph (dictionary)
       v: the advertiser node (string)
       b: the size of cluster b > k (integer)
       phi: the upper bound on the conductance (0 < phi < 1)
       unode: the user node lists
    """

    # constant parameters
    import logging
    logging.basicConfig(filename='message_running_050317_whole_sample_data.log' \
                        , level=logging.DEBUG)

    c1 = 200
    c3 = 1800
    c4 = 140

    degree = g.degrees

    mu_V = sum(degree)
    stat_dist = g.stationary_dist
    # TODO: Remove me
    # print("Start Step 1:\n")
    # start = time.time()
    # step 1

    l = int(np.ceil(np.log2(mu_V / 2.)))
    t1 = int(np.ceil(2. / phi ** 2 *
                     np.log(c1 * (l + 2) * np.sqrt(mu_V / 2.))))
    tLast = (l + 1) * t1
    epsilon = 1. / (c3 * (l + 2) * tLast * 2 ** b)
    # print("Step 1 used time %f\n" % (time.time() - start))

    # %%=======================================================================
    # start = time.time()
    # step 2
    # allVertices = np.array(list(g.Vertices()))
    # assuming that all the vertices are indexed from 0 to Len
    chi_v = sparse.coo_matrix(([1.], ([v], [0])), (g.len(), 1))
    q = chi_v.tocsr()

    r = Truncat(q, stat_dist * epsilon)
    # print q, r
    # print("Step 2 used time %f\n" % (time.time() - start))

    # %%=======================================================================
    # start = time.time()
    # step 3

    M = g.lazy_transition_mat.T.tocsr()
    # print("Step 3.4 used time %f\n" % (time.time() - start))

    if 2 ** b >= 5. / 6 * mu_V:
        print("""b (%f) is too large: 2**b > 5./6*mu_V. It should be less than %f""" %
              (b, np.log(5. / 6 * mu_V) / np.log(2)))
        return []
    # print("Step 3 used time %f\n" % (time.time() - start))

    # %%=======================================================================
    #    numOfLastCust = 0
    timeOfStayStill = 0
    numOfLastNode = 0



    for t in range(tLast):
        #if t % 1000 == 0:

        # logging.debug('running inside Nibble for t loop v is  %s  t is %s ',
        #               v,t)
        q = M.dot(r)
        r = Truncat(q, stat_dist * epsilon)

        # start = time.time()
        # get all the sorted none zero concentration nodes index
        # TODO: r might be better
        idx = ApproxSJ2(g, q)
        #
        numOfNode = len(idx)

        if numOfNode == numOfLastNode:
            timeOfStayStill += 1
        else:
            timeOfStayStill = 0
            numOfLastNode = numOfNode

        if timeOfStayStill > 10:  # no change after 10 times of iteration
            return idx

        # Condition C1
        #        if maxNumOfCust < k:
        #            continue

        #        allJ = np.where(numOfCust > k)
        if debug:
            f, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
            plt.ion()
            print("running inside Nibble for t loop v = %d; t = %d; len(idx) = %d" % (v, t, len(idx)))
        for j in range(1, len(idx)):  # allJ[0]:
            # start_j =time.time()
            idxJ = idx[:j]
            # lambdaJ = sum(out_dgreeU[idxJ] + in_degreeU[idxJ])/2.
            lambdaJ = sum(degree[idxJ])

            # condition C2 and C3
            if debug:
                ax2.plot(j, lambdaJ, 'ro')
                ax2.plot(j,  2 ** b,  'k+')
                plt.xlim([0, j+10])
            if not (lambdaJ <= 5. / 6 * mu_V and 2 ** b <= lambdaJ):
                continue

            # condition C1

            Phi_j = g.boundary_vol(idxJ)
            if debug:
                ax1.plot(j, Phi_j, 'ro')
                ax1.plot(j, phi, 'k+')
                plt.pause(1)

            if Phi_j > phi:
                continue
            # Condition C4
            qIdxJ = np.squeeze(np.asarray(q[idxJ].todense()))
            #lambdas = np.cumsum(out_dgreeU[idxJ] + in_degreeU[idxJ])/2.
            lambdas = np.cumsum(degree[idxJ])
            # the jj statisfies lambda_jj(q_t) <= 2**b <= lambda_jj+1(q_t)
            idxJJ = bisect.bisect(lambdas, 2 ** b)
            if idxJJ == 0:
                print("b is probably too small")
                return np.array([])

            # calculate Ix for each j using ranked q
            if idxJJ == len(qIdxJ):
                continue

            IxJ = qIdxJ[idxJJ] / stat_dist[idxJ][idxJJ]
            if debug:
                ax3.plot(j, IxJ, 'ro')
                ax3.plot(j, 1. / (c4 * (l + 2) * 2 ** b), 'k+')

            if IxJ >= 1. / (c4 * (l + 2) * 2 ** b):
                sJ = idxJ
                return sJ
            else:
                break  # this can be done as IxJ depends on k rather than j
                # end_j =time.time()
                # j_used_time = end_j - start_j
                # logging.debug('running inside Nibble v is %s j is %s t is %s j_used_time is %f', v,j ,t,j_used_time)
                #            if j > 10000 and j % 1000 == 0:
                #                print("running inside Nibble v is ", v, "j is ", j, "t is ", t)
    return np.array([])


if __name__ == "__main__":
    import pickle
    with open("../data/clean/order_no.pkl", 'rb') as f:
        order_no = pickle.load(f)
    with open("../data/clean/KHK_EAN.pkl", 'rb') as f:
        khk_ean = pickle.load(f)
    return_rate = pd.read_pickle("../data/clean/return_rate.pkl")
    with open("../data/clean/h_mat.pkl", 'rb') as f:
        h = pickle.load(f)
    bsk_label = pd.read_pickle("../data/clean/bsk_return_label.pkl")

    return_rate = return_rate.loc[khk_ean, :]
    bsk_label = bsk_label.loc[order_no, :]
    g = HyperGraph(order_no, khk_ean, return_rate['RET_Items'].values,
                   h, bsk_label['RET_Items'].values)

    b = 4
    phi = 0.5
    res = Nibble(g, 1, b, phi, True)
    print(res)
