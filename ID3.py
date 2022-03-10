from decicsion_tree import Tree, Classifier
from decicsion_tree import *
from experiments import *
"""
ID3 without any pruning

"""


def EX1_noPruning():
    tree = ID3_DecicsionTree()
    result = tree.fit()
    print(result.getAccuracy())






def EX3_PrunedDecicsionTree(M_param=1):
    tree = ID3_DecicsionTree(M_param, prune=True)
    result = tree.fit()
    # print(result.getAccuracy())
   
    ## Printing unpruned decicsion tree
    tree = ID3_DecicsionTree(M_param, prune=False)
    result = tree.fit()
    print(result.getAccuracy())


"""
ID3_PruningKfold:

Call the function with any array of parameters for early pruning with KFold
returns the best M value found
"""

def experiment(run_on_best_after_KFold=False):
    results = ID3_PruningKFold()
    best_acc_result = results[0]
    M = best_acc_result.M_param

    if run_on_best_after_KFold == True:
        tree = ID3_DecicsionTree(M, prune=True)
        result = tree.fit()
        print(result.getAccuracy)


def EX3_pruningWithKFold(M_params=[1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 80, 100, 120]): #### WITH GRAPH
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





if __name__ == "__main__":
    EX1_noPruning()
    pass


    '''EX 3 ID3 with pruning'''
    ########### Uncomment to run with pruning with chosen M_parameter (only prints ID3 without pruning)
    # EX3_PrunedDecicsionTree(M_param=1)



    ########### Uncomment to run with KFold
    ########### Finds the best M and can be called with 'run_on_best_after_KFold=True' to run ID3 with the parameter found
    ########### You can change the parameters in decicsion tree function ID3_PruningKFold  or use the other function below (EX3_pruningWithKfold to see the graph aswell and put your
    ########### parameters in there instead)

    # experiment()



    # You can also run EX3_pruningWithKFold to get the graphs at the end, i confused the name for experiemtn function and thought it was supposed to be a different file
    # EX3_pruningWithKFold(M_params=[1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 80, 100, 120])