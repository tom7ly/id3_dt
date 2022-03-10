from decicsion_tree import Tree, Classifier
from decicsion_tree import *
import matplotlib.pyplot as plt


def EX3_pruningWithKFold(M_params=[1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 80, 100, 120]):
    """
    function parameters:
        :param run_best_after_KFold
            set to True in order to run ID3 with optimized paramters after KFold

        :param M_params
            set to any array of int paramters, these are used the KFold for M
    """
    best_acc_result, _, accuracies = ID3_PruningKFold(M_params=M_params)
    M = best_acc_result.M_param

    accuracies_graph_line, M_graph_line = np.flip(accuracies[:, 4]), np.flip(accuracies[:, 0])
    plt.plot(M_graph_line, accuracies_graph_line)
    plt.ylabel("Accuracy")
    plt.xlabel("M_param")
    plt.grid(b=True)
    plt.show()
    return M


# def EX3_PrunedDecicsionTree(M_param=1):
#     tree = ID3_DecicsionTree(M_param, prune=True)
#     result = tree.fit()
#     # print(result.getAccuracy())

#     ## Printing unpruned decicsion tree
#     tree = ID3_DecicsionTree(M_param, prune=False)
#     result = tree.fit()
#     print(result.getAccuracy())

##==============================================================================================================

if __name__ == "__main__":
    
    # Uncomment EX3_pruningWithKfold(), you can put any parameters you want to run tests, put them as an array
    # as explained above

    EX3_pruningWithKFold()
    # EX3_PrunedDecicsionTree(M_param=1)
