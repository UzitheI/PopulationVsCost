import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
from public_tests import*
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
plt.title("Cost vs. Population per city")
plt.ylabel('Cost in $10,000')
plt.xlabel('Population of city in 10,000s')
plt.show()

#now we are going to draw a linear regression model to fit this data

def compute_cost(x,y,w,b):
    m=x.shape[0]
    total_cost=0
    for i in range(m):
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        total_cost = total_cost+ cost  
    total_cost = (1 / (2 * m)) * total_cost
    return total_cost

initial_w=2
initial_b=1

cost =compute_cost(x_train,y_train,initial_w,initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')
# compute_cost_test(compute_cost)

def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb= w*x[i]+b
        cost=f_wb-y[i]
        cost2=f_wb-y[i]
        cost2=cost2*x[i]
        dj_db=dj_db+cost
        dj_dw=dj_dw+cost2
    dj_db=(1/m)*dj_db
    dj_dw=(1/m)*dj_dw

    return dj_dw,dj_db

initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

compute_gradient_test(compute_gradient)

# test_w = 0.2
# test_b = 0.2
# tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

# print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing

# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)

#now plotting the data to find the best fit

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

# Create a scatter plot of the data. 
    
plt.plot(x_train, predicted, c = "b")
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Cost vs. Population per city")
# Set the y-axis label
plt.ylabel('Cost in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')

plt.show()

print('Here, we can now see a graph that shows the best linear fit for the model')

print('Now we can make essential predictions with the help of the model')

print('Give an appropriate value,ie 3.5=35000 or 1=10000')
print('------------------Make sure that the value is less than 22 (DataSet Restrictions)-----------------')
population=float(input())
predict = population * w + b
print('For population',population,' we predict a living cost of $%.2f' % (predict*10000))