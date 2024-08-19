#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[31]:


#CREATING ARRAYS

#1d numpy array (vector)
a = np.array([1,2,4],dtype = bool)
print(a)
print(type(a))
#2d numpy array (matrix)
b = np.array([[1,2,3],[4,5,6]], dtype= complex )
print(b)
#3d numpy array (tensor)
c = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype = float)
print(c)

#np.arange
np.arange(4,9,2)

#reshape arrays
np.arange(1,20,2).reshape(5,2)

#np.ones &np.zeros
np.ones((3,3))
np.zeros((3,3))

#np.random
np.random.random((4,4))

#np.linspace
d = np.linspace(-10,10,10, dtype= int)
print(d)

#np.identity
np.identity(4)


# In[45]:


#ARRAY ATTRIBUTES
a1 = np.arange(10 ,dtype = int)
a2 = np.arange(12, dtype = float).reshape(4,3)
a3 = np.arange(8).reshape(2,2,2)
a1.ndim
a3.shape
a2.size
a2.itemsize
a1.dtype


# In[46]:


#CHANGING DATATYPES
a3.astype(np.int64)


# In[14]:


#ARRAY OPERATIONS
b1 = np.arange(12).reshape(3,4)
b2 = np.arange(12,24).reshape(3,4)

#scalar operations
b1*2
b1+5
b1-2
b1/2
b1>5
b1==1

#vector operations
b1+b2
b1*b2
b1/b2
b1==b2


# In[45]:


#ARRAY FUNCTIONS
#MathematicalFunctions
c1 = np.random.random((3,3))
c1 = np.round(c1*100)
c2 = np.random.random((3,3))
c2 = np.round(c2*100)

print(c1)
np.max(c1)
np.min(c1)
np.product(c1)
np.min(c1,axis =0)

#Statistical Functions
np.mean(c1, axis= 0)
np.median(c1)
np.var(c1)
np.std(c1)

#Trigonometric Functions
np.sin(c1)

#Dot Product
np.dot(c1,c2)

#Log and exponent functions
np.log(c1)
np.exp(c1)

#Round,Floor and Ceil Functions
d1 = np.round(np.random.random((3,3))*100)
d2 = np.floor(np.random.random((3,3))*100)
d3 = np.ceil(np.random.random((3,3))*100)


# In[3]:


#INDEXING AND SLICING
e1 = np.arange(10)
print(e1)
e2= np.arange(12).reshape(3,4)
print(e2)
e3= np.arange(27).reshape(3,3,3)
print(e3)

#Indexing
#1d
e1[0]
e1[-1]
#2d
e2[1,2]
#3d
e3[1,1,0]

#Slicing
#1d
e1[2:5:2]
#2d
e2[:,2]
e2[1:, 1:3]
e2[0::2, 0::3]
e2[::2, 1::2]
e2[1::2, 0::3]
e2[0:2:, 1::]
e2[0:2, 1::2]
#3d
e3[1,0:,0:]
e3[0::2, 0:, 0:]
e3[0, 1, 0::]
e3[1,0::,1]
e3[2, 1::, 1::]
e3[0::2, 0:1:, 0::2]


# In[5]:


#ITERATING
for i in e2:
    print(i)

for i in np.nditer(e2):
    print(i)


# In[8]:


#RESHAPE
#Transpose
e2.T

#Ravel
e3.ravel()


# In[18]:


#STACKING
f1 = np.arange(12).reshape(6,2)
print(f1)
f2 = np.arange(12,24).reshape(6,2)
f2
#horizontal stacking
np.hstack((f1,f2))
#vertical stacking
np.vstack((f1,f2))


# In[17]:


#Splitting
np.hsplit(f1,2)
np.vsplit(f1,2)


# In[34]:


#ADVANCED INDEXING
g = np.arange(24).reshape(6,4)
print(g)
#fancy indexing
g[[0,2,3,5]]

#boolean indexing
g[g>15]  #greatewr than 15
g[g%2==0]  #even items
g[(g>10) & (g%2==0)] #greater than 10 and even (bitwise and is used because we are working with boolean)
g[g%7 !=0] #not divisible by 7


# In[44]:


#BROADCASTING
h1 = np.arange(12).reshape(4,3)
print(h1)
h2 = np.arange(3)
print(h2)
print(h1+h2) #broadcasting is taking place

i1 = np.arange(12).reshape(3,4)
print(i1)
i2 = np.arange(3)
print(i2)
print(i1+i2) #broadcasting can't take place


# In[67]:


#ANY MATHEMATICAL FUNCTION
def sigmoid(array):
    return 1/(1+np.exp(-(array)))
j = np.arange(10)
sigmoid(j)

actual = np.random.randint(1,50,10)
print(actual)
predicted = np.random.randint(1,50,10)
print(predicted )
def mse(actual,predicted):
    mse1= (actual-predicted)**2
    return np.mean(mse1)

print(mse(actual,predicted))


# In[73]:


#PLOTTING GRAPHS
import matplotlib.pyplot as plt
#plotting x = y (straight line)
x = np.linspace(-10.10,100)
y=x
plt.plot(x,y)


# In[74]:


#parabola
x = np.linspace(-10.10,100)
y = x**2
plt.plot(x,y)


# In[76]:


#sin curve
x = np.linspace(-10.10,100)
y= np.sin(x)
plt.plot(x,y)


# In[78]:


#xlog(x)
x = np.linspace(-10.10,100)
y = x* np.log(x)
plt.plot(x,y)


# In[80]:


#sigmoid
x = np.linspace(-10.10,100)
y = 1/(1+np.exp(-(x)))
plt.plot(x,y)


# In[97]:


#FUNCTIONS

k1 = np.random.randint(1,100,15)
np.sort(k1)
np.append(k1,999)


# In[98]:


k2 = np.random.randint(1,100,24).reshape(6,4)
k2
np.sort(k2, axis=0 ) #column wise
np.sort(k2, axis=1) #row wise
np.append(k2, np.ones((k2.shape[0],1)),axis=1)


# In[103]:


#concatenate
l1 = np.random.randint(1,100,24).reshape(6,4)
l2 = np.random.randint(1,100,24).reshape(6,4)
np.concatenate((l1,l2),axis = 1)


# In[107]:


#unique
m = np.array([1,1,1,3,2,4,4,4,9])
np.unique(m)
m.shape


# In[115]:


#expand dims
m2 = np.array([1,1,1,3,2,4,4,4,9])
m3 = np.expand_dims(m2,axis=1)
m3.shape


# In[127]:


#where
np.where(m2%2==0,0,m2)
#argmax
np.argmax(m2)
np.argmin(m2)


# In[134]:


#cumulative sum
n1 = np.random.randint(1,100,24)
np.cumsum(n1)
n2 = np.random.randint(1,100,24).reshape(6,4)
print(n2)
np.cumsum(n2, axis=0)

#cumulative product
n3 = np.random.randint(1,100,24)
np.cumprod(n3)
n4 = np.random.randint(1,100,24).reshape(6,4)
print(n4)
np.cumprod(n4, axis=0)


# In[144]:


#percentile
o1 = np.array([89 ,29 ,32 ,75 ,55 ,12 ,29 ,84 ,20, 7, 26, 26 ])
print(o1)
np.percentile(o1,50)


# In[145]:


#median 
np.median(o1)


# In[148]:


#histogram
np.histogram(o1, bins= [0,10,20,30,40,50])


# In[149]:


#Correlation coefficient
salary = np.array([2000,3000,5000,6000])
exp = np.array([1,2,3,4])
np.corrcoef(salary,exp)


# In[150]:


#isin
o1 = np.array([89 ,29 ,32 ,75 ,55 ,12 ,29 ,84 ,20, 7, 26, 26 ])
items = [8,29,7,26]
np.isin(o1,items)


# In[151]:


#flip
np.flip(o1)


# In[155]:


o2 = np.random.randint(1,100,24).reshape(6,4)
print(o2)
np.flip(o2)#both row and column flipping
np.flip(o2, axis=0)#only column flipping
np.flip(o2, axis=1)#only row


# In[159]:


#put
o3 = np.array([89 ,29 ,32 ,75 ,55 ,12 ,29 ,84 ,20, 7, 26, 26 ])
np.put(o3,[0,1],[900,3]) #permanent change
o3
#delete
np.delete(o3,[0,2,4])


# In[167]:


#set functions
p1 = np.array([1,2,3,4,5])
p2 = np.array([3,4,5,6,7])
np.union1d(p1,p2)
np.intersect1d(p1,p2)
np.setdiff1d(p1,p2) #items in p1 but not in p2
np.setxor1d(p1,p2) #removes common elements in p1 and p2
np.in1d(p1,3) #searches if an item is in the set or not


# In[169]:


#clip
a = np.array([89 ,29 ,32 ,75 ,55 ,12 ,29 ,84 ,20, 7, 26, 26 ])
np.clip(a, a_min= 15, a_max=80)

