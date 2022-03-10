from CostSensetiveID3 import P_params
from operator import sub
import matplotlib
from decicsion_tree import Tree, Classifier
import numpy as np
import sklearn
import pandas as pd
import matplotlib
from math import floor, sqrt
from typing import List
from sklearn import model_selection
import time
from numpy.random import default_rng as rng


class KNN_TreeResult:
    def __init__(self, res, subtree_accuracy, tree: Tree):
        self.fit_result = res
        self.tree = tree
        self.centroid = tree.centroid
        self.subtree_accuracy = subtree_accuracy


class KNN_Forest:
    def __init__(self, data: np.array, test_data: np.array, P: float, N: int, K: int, modified=True):
        self.train_data = data
        self.test_data = test_data
        self.P = P
        self.N = N
        self.K = K
        self.modified = modified
        self.forest = []

    def euclidean(self, sample1: np.array, sample2: np.array):
        size = sample1.size
        s = sample1[2]
        e = sample1[1]
        sum = 0
        for i in range(1, size):
            s1 = float(sample1[i]) ** 2
            s2 = float(sample2[i]) ** 2
            sum += abs(s1 - s2)

        return sqrt(sum)

    def KNN_testDataFromTrain(self, train_data: np.array, test_data: np.array, n_param: int):
        batch_size = floor(n_param * self.P)
        train_data_indices = np.random.randint(train_data.shape[0], size=batch_size)
        test_data_indices = []
        for k in range(train_data.shape[0]):
            if k not in train_data_indices:
                test_data_indices.append(k)
        test_data = train_data[test_data_indices, :]
        train_data = train_data[train_data_indices, :]
        return train_data, test_data

    def KNN_DecicsionTree(self, train_data: np.array, test_data: np.array, n_param: int) -> KNN_TreeResult:
        batch_size = floor(n_param * self.P)
        # train_data_indices = np.random.randint(train_data.shape[0], size=batch_size)
        train_data_indices = rng().choice(train_data.shape[0], size=batch_size, replace=False)
        temp_test_data_indices = np.delete(np.where(train_data[:, 0]), train_data_indices)

        sub_test_data = train_data[temp_test_data_indices, :]
        train_data = train_data[train_data_indices, :]
        num_of_attributes = train_data[0, :].size
        ids = range(1, num_of_attributes)
        # random_attr_indices = rng().choice(ids, replace=False, size=self.S)

        # train_data = np.delete(train_data, random_attr_indices, axis=1)
        # test_data = np.delete(test_data, random_attr_indices, axis=1)

        subtree = Tree(train_data=train_data, test_data=sub_test_data, toggleCentroid=True, prune=True)
        subtree_res = subtree.fit()
        M, P, R = 0, 0, 0
        if self.modified:
            tree = Tree(train_data, test_data, toggleCentroid=True, prune=True, M_param=30, P_param=8, R_param=0.3)
        else:
            tree = Tree(train_data, test_data, toggleCentroid=True, prune=True)
        res = tree.fit()

        res = KNN_TreeResult(res=res, tree=tree, subtree_accuracy=subtree_res)
        return res

    def KNN_Predict(self):
        test_data = self.test_data
        test_data_size = test_data[:, 0].size
        predictions = []
        for i, sample in enumerate(test_data):
            closest = []
            for j, tree in enumerate(self.forest):
                distance = self.euclidean(tree.centroid, sample)
                tree_accuracy = tree.subtree_accuracy.getAccuracy()
                if self.modified:
                    distance *= tree.subtree_accuracy.getAccuracy() * 100
                closest.append((distance, j, tree_accuracy))

            closest = np.array(closest)
            closest1 = closest[np.argsort(closest[:, 0])]
            closest2 = closest[np.argsort(closest[:, 2])]
            M_results = 0
            B_results = 0
            for tree_index, tree in enumerate(closest2[0 : self.K]):
                tree_index = int(tree[1])
                fit_result = self.forest[tree_index].fit_result
                tree = self.forest[tree_index].tree
                root = tree.root
                prediction = tree.perdict(node=root, sample=sample, parent=root)
                M_results += prediction == "M"
                B_results += prediction == "B"
            if M_results > B_results:
                predictions.append("M")
            else:
                predictions.append("B")
        correct_predictions = (predictions == test_data[:, 0]).sum()
        accuracy = correct_predictions / test_data_size
        # print(accuracy)
        return accuracy

    def KNN_Build(self):
        train_data, test_data = self.train_data, self.test_data
        n = train_data[:, 0].size
        for i in range(self.N):
            self.forest.append(self.KNN_DecicsionTree(train_data, test_data, n))

        accuracy = self.KNN_Predict()
        return accuracy


def KNN_InitKFoldForest(

    
    ## CHANGE PARAMS HERE
    ## ===================================================
    N_params=[12, 10, 12, 14, 16, 20, 32, 64, 128],
    P_params=[0.45, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
    K_params=[8, 4, 6, 14, 16, 20, 32, 64, 128],
    # S_params=[10, 4, 6, 8, 10, 15, 20],
    modified=False,
    ## ===================================================
):


    ## ===================================================

    train_data = pd.read_csv("train.csv").to_numpy()
    test_data = pd.read_csv("test.csv").to_numpy()
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=307943035)


    accuracies = []
    start = time.time()
    for n, N in enumerate(N_params):
        for p, P in enumerate(P_params):
            for k, K in enumerate(K_params):
                if K > N:
                    break

                acc_sum = 0
                for split in kf.split(train_data):
                    fold_train_data = train_data[split[0]]
                    fold_test_data = train_data[split[1]]
                    forest = KNN_Forest(
                        data=fold_train_data, test_data=fold_test_data, N=N, K=K, P=P, modified=modified
                    )
                    acc = forest.KNN_Build()
                    acc_sum += acc
                    print(acc)
                accuracy_avg = acc_sum / kf.get_n_splits()
                print(f"PROCESSING: N = [{N}]   ||   P = [{P}]   ||   K = [{K}]   ||   ACC = [{accuracy_avg}]")
                accuracies.append((N, K, P, accuracy_avg))
    end_time = time.time() - start
    print(accuracies)
    # print(accuracies[accuracies.index(np.amax(accuracies, axis=3))])
    # print(f"RUNTIME: {[end_time]}")


def KNN_InitKNNForest(N_forest_size=8, K_closest=4, P_param=0.3,modified=False):
    train_data = pd.read_csv("train.csv").to_numpy()
    test_data = pd.read_csv("test.csv").to_numpy()
    forest = KNN_Forest(data=train_data, test_data=test_data, N=N_forest_size, K=K_closest, P=P_param, modified=True)
    ACC = forest.KNN_Build()
    return ACC


# N_params = [30,16,32,64]
# K_params =[18,6,20,28]
# P_params=[0.45,0.4,0.5,0.6,0.7]
# results = []
# for n in N_params:
#     for k in K_params:
#         for p in P_params:
#             res = KNN_InitKNNForest(N_forest_size=n,K_closest=k,P_param=p,modified=True,ignore_attrs=0)
#             results.append((n,k,p,res))
#             print(f'N = {n} || K = {k} || P = {p} || ACCURACY = {res}')
# print(KNN_InitKNNForest(N_forest_size=10, K_closest=2, P_param=0.7, modified=False, ignore_attrs=0))

# KNN_InitKFoldForest(modified=False)

