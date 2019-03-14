# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import numpy as np
import pandas as pd
from lolviz import *
import lolviz
import math
from sklearn.metrics import confusion_matrix
from scipy import stats


class TreeNode:
    def __init__(self, value, p, q, left=None, right=None, n_nodes=1, cdepth = 0, ntype =""):
        self.value = value
        self.p = p
        self.q = q
        self.left = left
        self.right = right 
        self.n_nodes = n_nodes
        self.cdepth = cdepth
        self.ntype = ntype
        self.size = len(value)
        

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.hlim = math.ceil(math.log(sample_size,2))
        self.trees=None
        #self.path_length=None
                              

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if improved == True:
            jb=[stats.jarque_bera(X[:,i])[0] for i in range(X.shape[1])]
            X = X[:,np.where(jb > np.quantile(jb, 0.20))[0]]

        self.trees=[]
        for i in range(self.n_trees):
            random_indices = np.random.randint(0, X.shape[0], size=self.sample_size)  # generate N random indices
            sub_X = X[random_indices]  # get N samples with replacement
            itree = IsolationTree(self.hlim)
            individual = itree.fit(sub_X)
            self.trees.append(individual)
        return self.trees

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  
        """
        path_length_list=[]
        for obs in X:
            hx=0
            for tree in self.trees:
                hx += self.single_path(obs, tree)*1.0
            ehx = hx/self.n_trees
            path_length_list.append(ehx)
        return path_length_list

    def single_path(self, x, tree):
        if tree.ntype =='external':
            n=tree.size
            if n == 2: return tree.cdepth +1
            elif n > 2: return tree.cdepth + 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1)/n)
            else: return tree.cdepth +0
        else:
            a = tree.q
            tree.cdepth += 1
            if x[a] < tree.p:
                return self.single_path(x,tree.left)
            if x[a] >= tree.p:
                return self.single_path(x, tree.right)


    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        X = X.values
        hx = self.path_length(X)
        n = self.sample_size
        if n == 2: 
            deno = 1
        elif n > 2:
            deno = 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1)/n)
        else:
            deno = 0
            
        return 2**(-np.divide(hx,deno))
    
    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        ones = scores >= threshold
        prediction = ones.astype(int)
        return prediction



class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit 
        self.root = None
        self.p = None
        self.q = None
        self.nodes1 =0
    

    def fit(self, X:np.ndarray, improved=False, depth=0):
        

        Q = np.arange(np.shape(X)[1])
        if depth >= self.height_limit or len(X) <= 1 or (X == X[0]).all()==True:
            
            left=None
            right=None
            self.nodes1+=1
            self.root = TreeNode(X, self.p, self.q, left, right, n_nodes = self.nodes1, cdepth=depth+1, ntype='external')
            return self.root

        else:
            self.q = np.random.choice(Q)
            minimum = min(X[:,self.q])
            maximum = max(X[:,self.q])
            self.p = np.random.uniform(minimum, maximum)
            w = np.where(X[:,self.q] < self.p,True,False)
            X_left = X[w]
            X_right = X[~w]
            self.nodes1+=1

            return TreeNode(X, self.p, self.q, self.fit(X_left, depth=depth+1), self.fit(X_right, depth=depth+1), n_nodes = self.nodes1, cdepth=depth+1, ntype='internal')
                     
            
def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    thr = 1
    for i in range(10000):
        ones = scores >= thr        
        ypred = ones.astype(int)
        confusion = confusion_matrix(y, ypred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR:
            return thr, FPR
        else:
            thr-=0.01


