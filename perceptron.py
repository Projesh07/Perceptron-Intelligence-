#!/usr/bin/env python
# coding: utf-8

# In[227]:


import numpy as np
from matplotlib import pyplot

def perceptron(x1,x2,w1,w2,b):
    
    p=x1*w1+x2*w2+b
    return activation(p)

def activation(p):
    return 1/(1+np.exp(-p))


def cost(b):
    return activation(b) *( 1-activation(b))
    #return 2* (b-4)

def update(b_old):
    lrate=0.1
    b=b_old-lrate*error(b_old)
    return b

#input    x1   x2   target_output
dataset=[[3.0, 1.5, 1.0],
         [2.0, 1.0, 0.0],
         [4.0, 1.5, 1.0],
         [3.0, 1.0, 0.0],
         [3.5, 0.9, 1.0],
         [2.0, 0.5, 0.0],
         [5.5, 1.0, 1.0],
         [1.0, 1.0, 0.0],
        ]

test_data=[4.5,1]

# Random weight for input and bias

w1=np.random.randn()
w2=np.random.randn()
b=np.random.randn()

#ploting x y coridnate
xy_dimension=np.linspace(-20,20,100)
Y=activation(xy_dimension)
Y_cost=cost(xy_dimension)
pyplot.plot(xy_dimension,Y)
pyplot.plot(xy_dimension,Y_cost)


# In[228]:


#scater ploting the dataset
pyplot.axis([0,6,0,6])
pyplot.grid()
for i in range(len(dataset)):
    point=dataset[i]
    color='red'
    if point[2]==0:
        color='blue'
    pyplot.scatter(point[0],point[1],c=color)


# In[230]:


learning_rate=0.2
#tarinning perceptron
costs=[]
# Random weight for input and bias

w1=np.random.randn()
w2=np.random.randn()
b=np.random.randn()
for i in range(500000):
    rpi=np.random.randint(len(dataset)) #random point of the input
    point=dataset[rpi]
    perceptron=point[0]*w1+point[1]*w2+b
    prediction=activation(perceptron)
    
    target=point[2]
    _cost=np.square(prediction-target)
    
    costs.append(_cost)
    
    #print(point,initial_cost)
    #Update rule w=wo+learning_rate*(derivative of(w,b))
    der_of_cost_pred=2*(prediction-target)
    der_of_prediction_p=cost(perceptron)
    derz_of_w1=point[0]
    derz_of_w2=point[1]
    derz_of_b=1
    
    der_cost_w1=der_of_cost_pred*der_of_prediction_p*derz_of_w1
    der_cost_w2=der_of_cost_pred*der_of_prediction_p*derz_of_w2
    der_cost_b=der_of_cost_pred*der_of_prediction_p*derz_of_b
    
    
    w1=w1-learning_rate*der_cost_w1
    w2=w1-learning_rate*der_cost_w2
    b=b-learning_rate*der_cost_b
prediction_a=[]
for i in range(len(dataset)):
    point=dataset[i]
    perceptron=point[0]*w1+point[1]*w2+b
    prediction=activation(perceptron)
    prediction_a.append(prediction)
    print(point,prediction)

#pyplot.plot(prediction_a)


# In[ ]:





# In[ ]:




