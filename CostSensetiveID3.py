from decicsion_tree import Tree, Classifier
from decicsion_tree import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

P_params = ([0.5, 1, 2, 4, 6, 10, 12, 15, 20],)
R_params = ([0.002, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4],)


def EX4_DecicsionTreeLossNormal():
    """
    EX 4.1: 
    Normal version of the loss calculation without any improvement or pruning
    Prints loss value at the end

    """
    tree = ID3_DecicsionTree(prune=False)
    result = tree.fit()
    print(result.getLoss01())


def EX4_DecicsionTreeLossModified(
    M_params=[1], P_params=[1], R_params=[0],
):
    """
    EX 4.3
    Modified version with an improvement aims at achieving lower loss values.
    Best parameters found based on KFOLD were M=8,P=8, R=0.3
    
    Testing for different values (train and test sets):
        You can call this function with an array of values, in which case it will run on all of them and 
        print each result for the current itteration of input params.

    - Call the function with any paramters in the input arrays. 
    - For a single execution - run the function with 1 value in each input array.
    - Calling the function with printAll = True will print the loss for each value
    
    - To ignore any paramter, simply call the function without giving it any parametrs
        for example: if you want to call the function with only M_Params, call it like this:
            EX4_DecicsionTreeLossModified(M_params=[1, 2, 3, 5, 8, 16, 30, 50, 80, 120], printAll = False)
            This way P will use default value of 1 (to multiply the probability by 1)
            and R will also be ignored, meaning no early pruning will happen based on the amount of
            sick samples in the parent node.
):


    Parameters:
        :param M:
            used for early prune and early classify by majority
        :param P (Probability threshold):
            used as probability multiplier to reduce the IG value for attribute splits
            that have high probability of being classified as sick
            P is used to multiply the probability in the Entropy calculation function in decicsion_tree.py
        :param R (Ratio Threshold):
            this is the precentage of sick labels 'M', that any value above that label will result in
            an early pruning of the current node, labeling it as sick. The calculation is based on the labels by the father
            and is measured by the amount of samples present in the current node.

    """

    for M in M_params:
        for P in P_params:
            for R in R_params:
                tree = ID3_DecicsionTree(M=M, P=P, R=R, prune=True)
                result = tree.fit()
                print(result.getLoss01())


def EX4_KFoldForLoss(
    M_params=[1, 2, 3, 5, 8, 16, 30, 50, 80, 120],
    P_params=[0.5, 1, 2, 3, 6, 8, 10, 15, 20],
    R_params=[0.002, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
    modified=True,
):
    """

    EX 4.3 KFold
    KFold that aims on finding the best result best on loss value
    Call this function to get the best params based on the average of loss values.

    Call the function with param modified = True to enable KFold with modified version. aiming to minimize loss value.
    Ignore any params by simply calling the function without giving it any, for example:
    To ignore P,R simply call EX4_KfoldForLoss(M_params[1,2,...,], modified = True)

    :return best values for M,P,R based on KFold calculation
    """
    _, best_loss_result, accuracies = ID3_PruningKFold(M_params, P_params, R_params, modified=modified)
    M, P, R = best_loss_result.M_param, best_loss_result.P_param, best_loss_result.R_param
    loss_graph_line, M_graph_line = np.flip(accuracies[:, 3]), accuracies[:, 0]
    loss_graph_line = np.sort(loss_graph_line)
    M_graph_line=np.sort(M_graph_line)
    if modified:
        P_graph_line, R_graph_line = accuracies[:, 1], accuracies[:, 2]
        P_graph_line=np.sort(P_graph_line)
        R_graph_line=np.sort(R_graph_line)
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(loss_graph_line, M_graph_line, "o-")
        plt.ylabel("M_param")
        plt.xlabel("LOSS")

        plt.subplot(3, 1, 2)
        plt.plot(loss_graph_line, P_graph_line, "o-")
        plt.ylabel("R_param")
        plt.xlabel("LOSS")

        plt.subplot(3, 1, 3)
        plt.plot(loss_graph_line, R_graph_line, "o-")
        plt.ylabel("R_param")
        plt.xlabel("LOSS")

        plt.show()

    else:
        plt.plot(M_graph_line, loss_graph_line)
        plt.ylabel("Accuracy")
        plt.xlabel("M_param")

        plt.grid(b=True)
        plt.show()

    return M, P, R




if __name__ == "__main__":
    """Uncomment to run ID3  without improvement printing loss only"""
    # ---------------------------------------------------------------------------------------------------------------------
    # EX4_DecicsionTreeLossNormal()
    # ---------------------------------------------------------------------------------------------------------------------

    """Uncomment to run modified ID3 printing loss only, the values are set for the best params found by KFold"""
    # ---------------------------------------------------------------------------------------------------------------------
    EX4_DecicsionTreeLossModified(M_params=[8], P_params=[8], R_params=[0.3])  # BEST VALUES
    # EX4_DecicsionTreeLossModified(M_params=[1])                                # M param only example
    # ---------------------------------------------------------------------------------------------------------------------


    """If you want to test it with multiple values, use it like this:"""
    # ---------------------------------------------------------------------------------------------------------------------
    # EX4_DecicsionTreeLossModified(
    #     M_params=[1, 2, 3, 5, 8, 16, 30, 50, 80, 120],
    #     P_params=[0.5, 1, 2, 4, 6, 10, 12, 15, 20],
    #     R_params=[0.002, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
    # )
    # ---------------------------------------------------------------------------

    """ Run KFold for loss best results, returning M,P,K as a tuple.
    if modified is set to True, the algorithm will run the modified version utilizing P,R aswell
    set to False if you want to run an unmodified version without any improvement   
    """
    # ---------------------------------------------------------------------------------------------------------------------
    # best_params = EX4_KFoldForLoss(
    #     M_params=[1, 2, 4, 6, 8, 30, 50, 80],
    #     P_params=[0.5, 0.8, 1, 1.5, 2, 3, 4, 8],
    #     R_params=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
    #     modified=True,
    # )

    # ---------------------------------------------------------------------------------------------------------------------