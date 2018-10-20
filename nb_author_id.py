#!/usr/bin/python3

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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




#########################################################


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Choose classifier
clf = GaussianNB()
# Fit my data, so train using this classifier using the training features and training labels

t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

# Predict labels using the test data
t1 = time()
pred = clf.predict(features_test)
print("prediction time:", round(time()-t1, 3), "s")

# Compare prediction accuracy with test labels
print(accuracy_score(pred, labels_test))

# Cool, that's 97.3% accuracy! Ok, so what does this mean?!


#########################################################


