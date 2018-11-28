#!/usr/bin/python

"""
    starter code for the regression mini-project
    
    loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project)

    draws a little scatterplot of the training/testing data

    you fill in the regression code where indicated

"""    


import sys
import pickle
sys.path.append("/Users/kintesh/Documents/udacity_ml/python3/ud120-projects/tools")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("/Users/kintesh/Documents/udacity_ml/python3/ud120-projects/final_project/final_project_dataset_modified.pkl", "rb") )

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True, sort_keys = '/Users/kintesh/Documents/udacity_ml/python3/ud120-projects/tools/python2_lesson06_keys.pkl')
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color from "b" to "r"
### to differentiate training points from test points.


from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(feature_train, target_train)

print("Coefficient is: " + str(reg.coef_))
print("Intercept is: " + str(reg.intercept_))

pred = reg.predict(feature_train)

pred2 = pred.tolist()

print("-"*20)
print(target_train)
print("-"*20)
print(pred2)

print("Length of target: " + str(len(target_train)))
print("Length of prediction: " + str(len(pred2)))

#
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(target_train, pred2)
#
# print("Accuracy Score is: " + str(accuracy))

### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
