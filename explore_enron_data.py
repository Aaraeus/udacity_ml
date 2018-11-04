#!/usr/bin/python3

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("/Users/kintesh/Documents/udacity_ml/python3/ud120-projects/final_project/final_project_dataset.pkl", "rb"))

""" Okay, I think this is an array. So that means the length gives me the number of people in this dataset. """
print ("Size of data: " + str(len(enron_data)))

""" Okay, how do I see how many features we have? Let's try looking at the first element? """
""" NO WAIT! This is a dictionary, LOL. As per, you know, the comment above... ok, get all keys. These are our features. """

print("All the people in the Enron data: ")
for key in enron_data.keys():
    print(key)
print("End of people in Enron dataset.")

""" Okay so the keys are all the people. Find all keys for one dude, let's pick MARK METTS. """

print("Number of features: " + str(len(enron_data['METTS MARK'].keys())))

""" How many POIs are there? """

POI_count = 0
for key in enron_data.keys():
    if enron_data[key]["poi"] == 1:
        POI_count = POI_count + 1

print("Total number of POIs is " + str(POI_count))

""" Find total POIs from our text file """

f = open("/Users/kintesh/Documents/udacity_ml/python3/ud120-projects/final_project/poi_names.txt", "r")

contents = f.read()

print(contents)

""" Total value of stock for James Prentice """

""" First, gimme all of the keys associated to James pls. """

print("James' Keys: ")
for key in enron_data['PRENTICE JAMES'].keys():
    print(key)


print("James Prentice's stock value is: " + str(enron_data['PRENTICE JAMES']['total_stock_value']))

print("Wesley Colwell emails to a POI: " + str(enron_data['COLWELL WESLEY']['from_this_person_to_poi']))

print("Jeffry K Skilling's total stock options exercised: " + str(enron_data['SKILLING JEFFREY K']['exercised_stock_options']))

"""Of these three individuals (Lay, Skilling and Fastow), who took home the most money (largest value of “total_payments” feature)?"""

Payments_KL = enron_data['LAY KENNETH L']['total_payments']
Payments_JS = enron_data['SKILLING JEFFREY K']['total_payments']
Payments_AF = enron_data['FASTOW ANDREW S']['total_payments']

print("Total payments by the smartest men in the room: ")
print("Kenneth Lay: " + str(Payments_KL))
print("Jeffrey Skilling: " + str(Payments_JS))
print("Andrew Fastow: " + str(Payments_AF))

""" For nearly every person in the dataset, not every feature has a value. 
How is it denoted when a feature doesn’t have a well-defined value? """

print("")
print("Andrew Fastow Info:")
for key, value in enron_data['FASTOW ANDREW S'].items():
    print(key)
    print(value)

""" 
How many folks in this dataset have a quantified salary? 
What about a known email address? 
Salary key = 'salary'
email address = 'email_address'
"""

print("-" * 20)
count_error = 0
count_total = 0
for names, value in enron_data.items():
    count_total = count_total + 1
    if enron_data[names]['total_payments'] == 'NaN':
        count_error = count_error + 1
    # print("Salary for " + names + ": " + str(enron_data[names]['salary']))

print("Error for total_payments: " + str(count_error))
print("Quantified value for total_payments: " + str(count_total - count_error))
print("Proportion of total with NaN total payments: " + str((count_error) / count_total))


"""
How many POIs in the E+F dataset have “NaN” for their total payments? 
What percentage of POI’s as a whole is this?
"""

print("-" * 20)
count_error = 0
count_poi = 0
for names, value in enron_data.items():

    if enron_data[names]['poi'] == 1:
            count_poi = count_poi + 1
            if enron_data[names]['total_payments'] == 'NaN':
                count_error = count_error + 1
    # print("Salary for " + names + ": " + str(enron_data[names]['salary']))

print("Number of POIs: " + str(count_poi))
print("Proportion of POIs with invalid total payments: " + str((count_error) / count_poi))