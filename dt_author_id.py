#!/usr/bin/python3

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("/Users/kintesh/Documents/udacity_ml/python3/ud120-projects/tools")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
"""
QUESTION 1:
Using the starter code in decision_tree/dt_author_id.py, get a decision tree up and running as a classifier, 
setting min_samples_split=40. It will probably take a while to train. What’s the accuracy?
"""

from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split=40)

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(labels_test, pred)

print("-" * 40)
print("Accuracy for the model is: " + str(accuracy))
print("-" * 40)

""" QUESTION 2: 
Feature selection impact on performance.
How many features do we have?
"""

feature_count = len(features_train[0])
print("The number of features is:  " + str(feature_count))
print("Feature list is: ")
print(features_train[0])
print("-" * 40)

#########################################################


