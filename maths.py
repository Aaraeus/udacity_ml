"""
Just doing some maths calculations that I cba to get a scientific calculator for
"""

import math

""" Entropy and Information Gain """

entropy = -(2/3) * math.log(2/3, 2) - (1/3) * math.log(1/3,2)

print("Entropy is: " + str(entropy))

information_gain = 1 - ((3/4)*entropy + (1/4)*0)

print("Information gain is: " + str(information_gain))

x1 = 20
x2 = 80

# x1 = 40
# x2 = 80

# x1 = 60
# x2 = 80

# x1 = 20
# x2 = 100

# x1 = 20
# x2 = 120

y = 5 * x1 + 2.5 * x2 - 200

print(y)