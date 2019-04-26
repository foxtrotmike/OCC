# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:32:35 2019
A simple implementation of a nearest neighbor anamoly detector
if distance of a test example from its closest training example is larger than
the distance of the closest training example to its closest example in the training
set, then the test example is an outlier
@author: afsar
"""
import numpy as np
from sklearn.neighbors import KDTree

class NNAD:
    """
    Implementation of the nearest neighbor anamoly detector
    """
    def __init__(self, r = 0.0, k = 1):
        self.tree = None
        self.r = max(0,r)
        self.k = max(1,k)
    def fit(self,X):
        self.tree = KDTree(X)
        d,idx = self.tree.query(X,k=self.k+1)
        self.nnd = np.mean(d[:,1:],axis=1)
        return self
    def decision_function(self,X):
        d,idx = self.tree.query(X,k=self.k)
        return np.mean(d,axis=1)/(self.r+np.mean(self.nnd[idx],axis=1))-1
    
    
if __name__=='__main__':
    from plotit import plotit
    inputs = 4+np.random.randn(100,2)
    inputs = np.vstack((inputs,-4+np.random.randn(100,2)))    
    clf = NNAD().fit(inputs)
    plotit(inputs,clf = clf.decision_function,conts=[-1,0,1])