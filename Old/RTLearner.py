""""""
import random

"""  		  	   		 	   			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
"""

import numpy as np
import scipy.stats as sp

class RTLearner(object):
    """
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size = 1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "atung9"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        self.tree = self.build_tree(data_x,data_y)
        if self.verbose:
            print(self.tree.shape)
            print(self.tree[0:10][:])

    def predict(self, point, node):
        """
        Estimate a set of test points given the model we built.
        :param point: A numpy array representing a row from a set of points
        :return: The predicted result of the input data according to the trained model
        :rtype: float
        """
        #[factor,splitVal,left,right]
        node = int(node)

        if self.tree[node][0] == -1:
            return self.tree[node][1]

        val = point[int(self.tree[node][0])]
        if val <= self.tree[node][1]:
            return self.predict(point, node+self.tree[node][2])
        else:
            return self.predict(point, node+self.tree[node][3])

    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        y_pred = np.zeros(points.shape[0])
        for point in range(points.shape[0]):
            prediction = self.predict(points[point,:], 0)
            y_pred[point] = prediction
        return y_pred
    def find_feature(self, data_x, data_y):
        #find random feature to split on
        factor = np.random.randint(0,data_x.shape[1]-1)
        return factor

    def build_tree(self, data_x,data_y):
        # Pseudocode for decision tree algorithm based on slides from JR Quinlan
        # if	data.shape[0]	==	1:	return	[leaf,	data.y,	NA,	NA]
        # if	all	data.y same:	return	[leaf,	data.y,	NA,	NA]
        # else
        # determine	best	feature	i to	split	on
        # SplitVal =	data[:,i].median()
        # lefttree =	build_tree(data[data[:,i]<=SplitVal])
        # righttree =	build_tree(data[data[:,i]>SplitVal])
        # root	=	[i,	SplitVal,	1,	lefttree.shape[0]	+	1]
        # return	(append(root,	lefttree, righttree))

        #node structure is (factor,splitval,left,right)
        if data_x.shape[0] <= self.leaf_size:
            #return leaf node if less than leaf size get the mean of remaining values
            if data_y.size == 0:
                return np.array([[-1, 0, np.nan, np.nan]])
            else:
                return np.array([[-1, sp.mode(data_y)[0][0], np.nan, np.nan]]) #changed to mode instead of mean to classify instead of regression
        if np.all(np.isclose(data_y,data_y[0])):  #reference: https://numpy.org/doc/stable/reference/generated/numpy.all.html
            return np.array([[-1, data_y[0], np.nan, np.nan]])
        #find random feature to split
        rand_feature = self.find_feature(data_x,data_y)
        splitVal = np.median(data_x[:, rand_feature])

        #continue building tree recursively
        left = data_x[:, rand_feature] <= splitVal
        right = data_x[:, rand_feature] > splitVal
        # if self.verbose:
        #     print(left)

        #infinite recursion case, split by mean instead of median
        if np.array_equal(data_x[left], data_x): #check if left split data is equal to current data, causing infinite recursion reference: https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
            splitVal = np.mean(data_x[:, rand_feature])
            left = data_x[:, rand_feature] <= splitVal
            right = data_x[:, rand_feature] > splitVal

        leftTree = self.build_tree(data_x[left], data_y[left])
        rightTree = self.build_tree(data_x[right], data_y[right])

        root = np.array([[rand_feature, splitVal, 1, leftTree.shape[0] + 1]])
        #return node
        DTArray = np.vstack((root, leftTree, rightTree))
        return DTArray


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")