from numpy.core.fromnumeric import amax, amin, shape
from numpy.f2py.crackfortran import true_intent_list
import sklearn
from sklearn import model_selection
from numpy import log2 as log
from numpy.core.defchararray import add, rpartition
from pandas.core.frame import DataFrame
import pandas as pd
from sklearn.metrics import confusion_matrix
import sklearn
import sys
import numpy as np
from copy import copy
import time
from math import floor

eps = np.finfo(float).eps


class ID3_Result:
    def __init__(self, accuracy=None, loss=None, M_param=1, R_param=None, P_param=None):
        self.accuracy = accuracy
        self.loss = loss
        self.M_param = M_param
        self.R_param = R_param
        self.P_param = P_param


def printResults(results):
    best_accuracy_index = results.index(np.amax(results, axis=3))
    accuracy_based_result = results[best_accuracy_index]
    best_loss_index = results.index(np.amin(results, axis=2))
    best_loss_result = results[best_loss_index]


def printAccuracies(accuracies):
    for i, acc in enumerate(accuracies):
        print(str(f"SPLIT: {acc[0]} || M_PARAM: {acc[1]} || ACCURACY: {str(acc[2])}"))


def printLosses(losses):
    for i, loss in enumerate(losses):
        print(str(f"SPLIT: {loss[0]} || M_PARAM: {loss[1]} || LOSS: {str(loss[2])}"))


class Classifier:
    def __init__(self, total_tests, correct_predictions, true_positive, true_negative, false_positive, false_negative):
        self.total_tests = total_tests
        self.correct_predicitons = correct_predictions
        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative

    def getAccuracy(self):
        return self.correct_predicitons / self.total_tests

    def getLoss01(self):
        return ((self.false_positive * 0.1) + self.false_negative) / self.total_tests

    def getParams(self):
        return (
            self.total_tests,
            self.correct_predicitons,
            self.true_positive,
            self.true_negative,
            self.false_positive,
            self.false_negative,
        )


class Tree:
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        M_param=1,
        P_param=1,
        R_param=0,
        toggleCentroid=False,
        prune=False,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.labels = self.train_data[:, 0]
        self.predictions = []
        self.tree = dict()
        self.splits = dict()
        self.splits_thresh = dict()
        self.M_param = M_param
        self.prob_multiplier = P_param
        self.R_param = R_param
        self.toggleCentroid = toggleCentroid
        self.prune = prune

        parent = self.Node(data=self.train_data)
        self.root = self.ID3(data=self.train_data, parent=parent)

    class Node:
        def __init__(self, data: np.array, threshold=0, splitter=0, left=None, right=None):
            self.data = data
            self.threshold = threshold
            self.splitter = splitter
            self.left = left
            self.right = right
            self.classification = None

    def getAttributeThresholds(self, attribute_values: np.array):
        thresholds = []
        attribute_values = np.sort(attribute_values)
        attribute_values = np.unique(attribute_values)
        for i, val in enumerate(attribute_values[:-1]):
            thresh = (val + attribute_values[i + 1]) / 2
            thresholds.append(thresh)
        return thresholds

    def getEntropy(self, attribute_values: np.array, labels: np.array, size) -> float:
        M_probability, B_probability = self.getProbs(attribute_values, labels, size)
        H_M = 0 if M_probability == 0 else M_probability * log(M_probability)
        H_B = 0 if B_probability == 0 else B_probability * log(B_probability)
        H = H_M + H_B
        return -H

    def getProbs(self, attribute_values: np.array, labels: np.array, size):

        M_size, B_size = 0, 0
        M_size, B_size = labels[labels == "M"].size, labels[labels == "B"].size
        if size == 0:
            return 0
        return (self.prob_multiplier * (M_size / size + eps)), B_size / (size + eps)

    def Split(self, attribute_values: pd.DataFrame, threshold: float):

        right_split = np.where(attribute_values >= threshold)
        left_split = np.where(attribute_values < threshold)

        return left_split, right_split

    def getIG(
        self,
        node_data: np.array,
        attribute_idx: int,
        current_node_labels: np.array,
        left_indices: np.array,
        right_indices: np.array,
    ):
        size, size_left, size_right = len(node_data[:, attribute_idx]), len(left_indices[0]), len(right_indices[0])
        attr_data = node_data[:, attribute_idx]
        left_labels = current_node_labels[left_indices]
        right_labels = current_node_labels[right_indices]
        attr_data_left = node_data[left_indices][:, attribute_idx]
        attr_data_right = node_data[right_indices][:, attribute_idx]
        H_entropy = self.getEntropy(attr_data, current_node_labels, size)
        H_left = self.getEntropy(attr_data_left, left_labels, size_left)
        H_right = self.getEntropy(attr_data_right, right_labels, size_right)
        fraction_left, fraction_right = size_left / size, size_right / size
        H_left, H_right = H_left * fraction_left, H_right * fraction_right

        gain = H_entropy - H_left - H_right

        return gain

    # def getGini(self, attribute_values, labels, size):
    #     M_probability, B_probability = self.getProbs(attribute_values, labels, size)
    #     return 1 - (M_probability ** 2) - (B_probability ** 2)

    def getAttributeThresholdAndIG(self, node_data: np.array, attribute_idx: int, current_node_labels: np.array):

        thresholds = self.getAttributeThresholds(node_data[:, attribute_idx])
        max_gain = 0
        best_gini = 1
        max_threshold = 0
        gini_threshold = 0
        for idx, t in enumerate(thresholds):
            left_split_indices, right_split_indices = self.Split(node_data[:, attribute_idx], t)
            gain = self.getIG(node_data, attribute_idx, current_node_labels, left_split_indices, right_split_indices)
            if gain >= max_gain:
                max_gain = gain
                max_threshold = t

        return max_gain, max_threshold

    def DecideBasedOnMajority(self, M_labels: pd.DataFrame, B_labels=pd.DataFrame):
        M_size = len(M_labels[0])
        B_size = len(B_labels[0])
        return "M" if M_size > B_size else "B"

    def checkEndCondition(self, data: np.array, parent_data: np.array):
        # print(data)
        # m = data[:,data == "M"]
        m = np.where(parent_data[:, 0] == "M")
        b = np.where(parent_data[:, 0] == "B")
        s = data.data[:, 0].size

        if self.prune:
            if s != 0:
                frac = len(m) / s
                if frac >= self.R_param and self.R_param != 0:
                    return "M"

        if s < self.M_param and self.prune:
            # prune by majority
            return self.DecideBasedOnMajority(m, b)
        if len(m[0]) != 0 and len(b[0]) != 0:
            return False

        if len(m[0]) == 0:
            return "B"

        return "M"

    def getMaxGainIdx(self, data: np.array, attributes_gains: np.array):
        split_index_num = 0
        max_gain = max(attributes_gains)
        idx = attributes_gains.index(max_gain)
        for idx, maxval in enumerate(attributes_gains):
            if maxval == max_gain:
                split_index_num = idx
        return split_index_num

    # def getBestGiniIdx(self, data: np.array, attribute_ginis: np.array):
    #     split_index_num = 0
    #     best_gini = min(attribute_ginis)
    #     idx = attribute_ginis.index(best_gini)
    #     for idx, minval in enumerate(attribute_ginis):
    #         if minval == best_gini:
    #             split_index_num = idx
    #     return split_index_num

    def ID3(self, data: np.array, parent: Node):
        node = self.Node(data=data)
        node.parent = parent
        res = self.checkEndCondition(node, parent_data=parent.data)
        if res != False:
            self.left = self.right = None
            node.classification = res
            return node

        attributes_size = data[0].size
        labels = data[:, 0]
        attributes_gains = []
        attributes_thresholds = []
        attributes_gains.append(-1)
        attributes_thresholds.append(-1)

        for i in range(1, attributes_size):
            gain, thresh = self.getAttributeThresholdAndIG(data, i, labels)
            attributes_gains.append(gain)
            attributes_thresholds.append(thresh)

        # GET BY IG
        split_index_num = self.getMaxGainIdx(data, attributes_gains)
        selected_threshold = attributes_thresholds[split_index_num]

        left_split_idx = np.where(data[:, split_index_num] < selected_threshold)
        right_split_idx = np.where(data[:, split_index_num] >= selected_threshold)
        node.threshold = selected_threshold
        node.splitter = split_index_num
        left_data = data[left_split_idx]
        right_data = data[right_split_idx]
        node.left = self.ID3(left_data, parent=node)
        node.right = self.ID3(right_data, parent=node)
        return node

    # def buildTree(self) -> Node:
    #     parent = self.Node(data=self.train_data)
    #     self.root = self.ID3(data=self.train_data, parent=parent)

    #     return self.root

    ## ============================================================================================================================================================
    ## ============================================================================================================================================================

    def perdict(self, node: Node, sample, parent: Node):
        res = self.checkEndCondition(node, parent_data=parent.data)
        if res != False:
            return res

        current_node_splitter = node.splitter
        current_node_thresh = node.threshold
        f = sample[current_node_splitter]
        if sample[current_node_splitter] > current_node_thresh:
            return self.perdict(node.right, sample, node)

        return self.perdict(node.left, sample, node)

    def fit(self, samples=None, root=None) -> Classifier:
        if root is None:
            root = self.root
        if samples is None:
            samples = self.test_data
        self.predictions = []
        for sample in samples:
            prediction = self.perdict(node=root, sample=sample, parent=root)
            self.predictions.append(prediction)

        if self.toggleCentroid == True:
            self.setCentroid()
        return self.checkAccuracy(samples)

    def setCentroid(self):
        samples = self.train_data
        num_of_features = samples[0, :].size
        num_of_samples = samples[:, 0].size
        centroid = []
        centroid.append("centeroid")
        for feature in range(1, num_of_features):
            featureAvg = 0
            featureAvg = samples[:, feature].sum() / num_of_samples
            centroid.append(featureAvg)
        self.centroid = np.array(centroid)

    def checkAccuracy(self, samples: np.array) -> Classifier:
        target = samples[:, 0]
        total = len(target)
        fp, fn, = 0, 0
        tp, tn = 0, 0
        tp_tn = (self.predictions == target).sum()
        for i, prediction in enumerate(self.predictions):
            if prediction == "M":
                if prediction != target[i]:
                    fp += 1
                else:
                    tp += 1
            elif prediction == "B":
                if prediction != target[i]:
                    fn += 1
                else:
                    tn += 1

        result = Classifier(
            total_tests=total,
            correct_predictions=tp_tn,
            true_positive=tp,
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
        )
        return result

    def lossFunction01(self, total, true_pos, true_neg, false_pos, false_neg):
        return ((0.1 * (false_pos)) + false_neg) / (total)


## ============================================================================================================================================================
## ============================================================================================================================================================
def ID3_DecicsionTree(train_data=None, test_data=None, M=1, P=1, R=0, prune=False, loss_only=False) -> Tree:
    if train_data is None or test_data is None:
        train_data = pd.read_csv("train.csv").to_numpy()
        test_data = pd.read_csv("test.csv").to_numpy()

    idt = Tree(train_data=train_data, test_data=test_data, P_param=P, M_param=M, R_param=R, prune=prune)
    return idt
    # results = idt.fit()

    # if loss_only == True:
    #     print(results.getLoss01())
    # else:
    #     print(results.getAccuracy())


def ID3_PruningKFold(
    M_params=[1, 2, 3, 5, 8, 16, 30, 50, 80, 120],
    P_params=[0.5, 1, 2, 4, 6, 10, 12, 15, 20],
    R_params=[0.002, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
    modified=False,
):
    train_data = pd.read_csv("train.csv").to_numpy()
    test_data = pd.read_csv("test.csv").to_numpy()
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=307943035)

    if modified == False:
        P_params = [1]
        R_params = [0]

    accuracies = []
    
    for m, M in enumerate(M_params):
        for p, P in enumerate(P_params):
            for r, R in enumerate(R_params):
                # print(f'R = {R} || P = {P} || M = {M}')
                accuracy_sum = 0
                loss_sum = 0
                for s, split in enumerate(kf.split(train_data)):
                    fold_train = train_data[split[0]]
                    fold_test = train_data[split[1]]
                    # fold = Tree(train_data=fold_train,test_data=fold_test,M_param=M,prob_multiplier=P)
                    fold = ID3_DecicsionTree(fold_train, fold_test, M, P, R, prune=True)
                    res = fold.fit()
                    acc = res.getAccuracy()
                    accuracy_sum += acc
                    loss = res.getLoss01()
                    loss_sum += loss
                    # print(f"M={M}|P={P}|R={R} | LOSS: {loss}, ACC: {acc}")
                acc_avg = accuracy_sum / kf.get_n_splits()
                loss_avg = loss_sum / kf.get_n_splits()
                # print(acc_avg)
                accuracies.append((M, P, R, loss_avg, acc_avg))

    # for acc in accuracies:
    #     print(f"M={acc[0]} | P={acc[1]} | R={acc[2]} || ACCURACY = {acc[4]}")
    accuracies = np.array(accuracies)
    best_acc_result, best_loss_result = ID3_getBestParams(accuracies)

    return best_acc_result, best_loss_result, accuracies


def ID3_getBestParams(results, modified=False):  # SET TO TRUE INORDER TO FIND BEST PARAMS FOR MINIMAL LOSS
    pd.DataFrame(results).to_csv("file.csv")
    best_acc = 0
    best_loss = 100
    accres = None
    lossres = None
    for res in results:
        if res[4] >= best_acc:
            best_acc = res[4]
            accres = res
        if res[3] <= best_loss:
            best_loss = res[3]
            lossres = res

    M, P, K, loss, acc = accres
    best_accuracy_result = ID3_Result(accuracy=acc, loss=loss, M_param=M, P_param=P, R_param=K)
    M, P, K, loss, acc = lossres
    best_loss_result = ID3_Result(accuracy=acc, loss=loss, M_param=M, P_param=P, R_param=K)
    return best_accuracy_result, best_loss_result




















# def ID3_WithoutPruning(M=1, P=1, R=0) -> Tree:

#     tree = ID3_DecicsionTree(M=M, P=P, R=R, prune=False)
#     result = tree.fit(prune=True)
#     accuracy = result.getAccuracy()
#     loss = result.getLoss01()
#     print(f"ACCURACY = {accuracy}\n LOSS = {loss}")
#     return tree


# ID3_DecicsionTreePruning(M)


# print("======================================\nDONE")
# DT_WithPruning(True)
# tree = ID3_DecicsionTree()
# f = tree.fit()
# print(f.getAccuracy())
# print(f.getLoss01())
# DT_WithPruning(modified=True)
# DT_WithoutPruning(M=8, P=8, R=0.3)
