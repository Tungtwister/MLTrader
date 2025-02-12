""""""
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


class BagLearner(object):
    """
    This is a Bag Learner.
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        """
        Constructor method
        """
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        #code from assignment to instantiate several learners
        self.learners = []
        for i in range(0, self.bags):
            self.learners.append(learner(**self.kwargs))

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "atung9"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        #for each learner create a bag 100% of original data, but sample with replacement so that n' can be less than n
        for learner in self.learners:
            bag = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True) #reference: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
            if self.verbose:
                print(bag)
            learner.add_evidence(data_x[bag,:], data_y[bag])

    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        Y = np.zeros((self.bags,points.shape[0]))
        for idx, learner in enumerate(self.learners):
            Y[idx,:] = learner.query(points)
            if self.verbose:
                print(Y[idx,:])

        pred_y, count = sp.mode(Y, axis=0) #calculate using mode instead of mean to be classification learner
        return pred_y


if __name__ == "__main__":
    pass