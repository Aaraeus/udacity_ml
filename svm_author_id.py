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
# from class_vis import prettyPicture
from sklearn.metrics import accuracy_score


# Minimise data to 1% of original size to see accuracy difference
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

print("-"*10)
print("SIZES OF DATA")
print("-"*10)
print("Features Train:")
print(features_train.shape)
print("Features Test:")
print(features_test.shape)
print("Labels Train:")
print(len(labels_train))
print("Labels Test:")
print(len(labels_test))
print("-"*10)

# Choose classifier
clf = svm.SVC(kernel="linear", C=1, gamma=1)

# fit the model using training data
t0 = time()
clf.fit(features_train, labels_train)
print("Training time:", round(time()-t0, 3), "s")

# predict based on training data
t1 = time()
pred = clf.predict(features_test)
print("Prediction time:", round(time()-t1, 3), "s")

# Show pic!
# prettyPicture(clf, features_train, features_test)


# Use accuracy score to determine accuracy
print("-"*10)
print("Accuracy score: ")
print(accuracy_score(pred, labels_test))
print("-"*10)

#########################################################


