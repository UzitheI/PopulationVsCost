import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
# %matplotlib inline
x_train,y_train=load_data();
#loading the data set from the function in utils.py
print("Type of x_train:",type(x_train))
print("First 5 elements of x train are :\n",x_train[:5])

print("Type of y_train:",type(y_train))
print("First 5 elements of y_train are:\n",y_train[:5])
#showing the first 5 elements of the data set 

#to check the dimension of the variable 
print('The shape of x-train is:',x_train.shape)
print('The shape of y-train is:',y_train.shape)
#shows that it is a 1D array
print('The number of training examples are :',len(x_train))

#now drawing a scatter plot to visualize the data

plt.scatter(x_train,y_train,marker='*',c='r')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of city in 10,000s')
plt.show()

#now we are going to draw a linear regression model to fit this data

def compute_cost(x,y,w,b):
    m=x.shape[0]
    total_cost=0
    for i in range(m):
        total_cost+=math.pow((w*x[i]+b)-y[i],2)
        total_cost=total_cost/(2*m)

    return total_cost

initial_w=2
initial_b=1

cost =compute_cost(x_train,y_train,initial_w,initial_b)
print(type(cost))
print(f'Cost at initial w:{cost:.3f}')

from public_tests import*
compute_cost_test(compute_cost)