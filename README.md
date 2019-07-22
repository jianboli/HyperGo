# HyperGo
HyperGo is a return forecast algorithm developed in our paper [E-tail product return prediction via hypergraph-based local graph cut](https://www.kdd.org/kdd2018/accepted-papers/view/e-tail-product-return-prediction-via-hypergraph-based-local-graph-cut) (KDD 2018). A vedio summary of the paper can be find here: https://www.youtube.com/watch?v=SFI3sc_K6ao

# Requirement
This code runs under python 3+ and depends on the following libraries:

1. numpy
2. pandas
3. scipy
4. pickel
5. sklearn

# Note
As the return data studied in this project is private, we cannot release it. Here, we simulate data to demonstrate how to use of the code. 

# To run the code
The code can be run with two steps:
1. Run the Simulate data.ipynb under notebook to generate simulate data and to understand the data structure
2. This step can be run in two ways:
    * first run the pre_process.py and then run the main.py to run the hypergo
    * simply run any of the main_*.py to do the studies used inside the paper. The main scripts will random splits the data and then tain, test etc. 
    It should be relatively straightforward to know the purpose of each main script based on the name.
    
Some of the postprocessing scripts are also included under the script folder.

# TODO
The codes have been cleaned and tested. You should be able to run through the code without any issues. If you meet any issues, please leave me a message.

Thanks and enjoy. 
