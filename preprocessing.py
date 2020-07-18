# import relevant files packages for data preprocessing  ...

import pandas as pd
import numpy as np
import os, re


class LoadPrep(object):
    """
    Load and process the data into usable format
    Parameters:
        - file or filepath to read a csv/txt file
    """

    def __init__(self, dataframe): # defines the constructor:
        self.dataframe = dataframe

    def get_matrices(self, print_enabled=True):
        """returns the adjacency matrix of entries in a dataframe:"""
        self.df = pd.read_csv(self.dataframe)

        self.columns = set.union(set(self.df.Va_Name),set(self.df.Vi_Name))
        self.index = set.union(set(self.df.Va_Name),set(self.df.Vi_Name))
        sr_amt =pd.DataFrame(np.zeros(shape=(len(self.index),len(self.columns))),columns=self.columns,\
                             index=self.index)
        for v1, v2, r in zip(self.df.Va_Name, self.df.Vi_Name, self.df.Edge): # update values ...
            if r>0:
                sr_amt.at[v1,v2]=r
            else:
                sr_amt.at[v1,v2]=0
        #actual datafroame: sr_amt
        if print_enabled:
            print('Relevant matrices from %s  file ...'%(self.dataframe))

        # return ndarray of adjacency matrix:
        adj_mat = np.array(sr_amt)
        # diagonal matrix: why not make the matrices in different functions?
        diag_mat = np.diag(adj_mat.sum(axis=1))
        # laplacian matrix:
        lap_mat = diag_mat - adj_mat

        return adj_mat, diag_mat, lap_mat

    def interaction_mat(self):
        """ interaction intensity a matrix of nodes vs.nodes and the entres as count or
        frequency of mentioning communities """
        columns = self.columns # use variables declared in the previous function
        index = self.index
        mentioning_df = pd.DataFrame(np.zeros(shape=(len(index),len(columns))),columns=columns, index=index) #
        # update with relevant values ...
        for v1, v2, e, c in zip(self.df.Va_Name, self.df.Vi_Name, self.df.Edge, self.df.Count):
            if e>0: # check if an edge exists between the users
                mentioning_df.at[v1,v2]=c # update with the count of interaction
            else:
                mentioning_df.at[v1,v2]=0
        # get the users with pairwise mentioning only #p=sr_vrcom[sr_vrcom.values>0]

        return np.array(mentioning_df)
