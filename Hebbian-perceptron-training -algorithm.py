#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
from matplotlib import pyplot

def hebbian_perceptron(x1,x2,w1,w2,b):
    
    p=x1*w1+x2*w2+b
    return activation(p)

def activation(p):
    if p>=1.0:
        return 1.0
    else:
        return 0.0


def cost(predict,target):
    if predict==target:
        return target
    else:
        return 0.0
    


#input x1  x2  target_output
dataset=[[1.0, 1.0, 1.0],
         [1.0, 0.0, 1.0],
         [0.0, 1.0, 1.0],
         [0.0, 0.0, 0.0],
        ]



# In[37]:





# In[106]:


learning_rate=0.5
#tarinning perceptron
costs=[]
# Random weight for input and bias

w1=0.0
w2=0.0
b=-1
for i in range(len(dataset)):
    #rpi=np.random.randint(len(dataset)) #random point of the input
    #print(rpi)
    point=dataset[i]
    perceptron=point[0]*w1+point[1]*w2+b
    prediction=activation(point[2])
        
    w1=w1+learning_rate*point[0]*prediction
    w2=w2+learning_rate*point[1]*prediction
    b=b+learning_rate*prediction

#pyplot.plot(prediction_a)
test_data=[-1,0]
preceptron=test_data[0]*w1+test_data[1]*w2+b
prediction=activation(preceptron)
print(w1,w2,b)
print(preceptron)
prediction

#scater ploting the dataset
pyplot.axis([-6,6,-6,6])
pyplot.grid()
for i in range(len(dataset)):
    point=dataset[i]
    color='red'
    if point[2]==0:
        color='blue'
    pyplot.scatter(point[0],point[1],c=color)
if(prediction>0.5):
    
    print("On")
    #scater ploting the dataset
    pyplot.scatter(test_data[0],test_data[1],c="red")
else:
    print("Off")
    pyplot.scatter(test_data[0],test_data[1],c="blue")
pyplot.plot()


# In[ ]:




