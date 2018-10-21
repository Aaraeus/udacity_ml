#!/usr/bin/python3

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append(r"C:\Users\Kintesh\Desktop\code\udacity_ml\ud120-projects-master\tools")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### make sure you use // when dividing for integer division


#########################################################
from sklearn import svm
from class_vis import prettyPicture
from sklearn.metrics import accuracy_score

# Choose classifier
clf = svm.SVC(kernel="linear", C=1, gamma=1)

# fit the model using training data
clf.fit(features_train, labels_train)

# predict based on training data
pred = clf.predict(features_test)

# Show pic!
prettyPicture(clf, features_train, labels_train)


# Use accuracy score to determine accuracy
print(accuracy_score(pred, labels_test))

#########################################################


