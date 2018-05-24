import pickle
import pandas as pd
# from scoop import futures

from src.pre_process_cls import SplitTrainValidateTest
from src.baseline_prediction_cls import BaseLinePredictor
from src.knn_prediction_cls import knn_Predictor
from src.hg_prediction_2_step_cls import HypergraphPredictor

input_path = 'data/clean/'
output_path = 'data/split_'
result_path = 'rst'
seeds = range(1)

split = SplitTrainValidateTest(0.6, 0.2)
split.read_data(input_path)

tot_size = []
tot_time = []
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

    # hyper graph ====================================
    p = HypergraphPredictor(max_num_test=300, parallel="Multi", n_cpu=7, chunk_size=1)
    p.fit(h_train, bsk_label_train, order_no_train, khk_ean_train,
          return_rate_train, r_train, multi_item_bsk, ratio=None, step=2)

    s, t = p.timing(h_validate, bsk_label_validate, order_no_validate, r_validate,
                    [6, 7, 8, 9, 10],
                    [0.8, 0.6, 0.4, 0.3])
    tot_size += s
    tot_time += t

timing = pd.DataFrame({'size':tot_size,  'time':tot_time})
timing.to_csv(result_path + "/hypergraph_time_complex.csv")



