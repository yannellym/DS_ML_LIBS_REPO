#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # NumPy Exercises 
# 
# Now that we've learned about NumPy let's test your knowledge. We'll start off with a few simple tasks, and then you'll be asked some more complicated questions.

# #### Import NumPy as np

# In[1]:


import numpy as np


# #### Create an array of 10 zeros 

# In[3]:


np.zeros(10)


# #### Create an array of 10 ones

# In[4]:


np.ones(10)


# #### Create an array of 10 fives

# In[14]:


np.array([5]*10)


# #### Create an array of the integers from 10 to 50

# In[15]:


np.arange(10,51)


# #### Create an array of all the even integers from 10 to 50

# In[16]:


np.arange(10, 51, 2)


# #### Create a 3x3 matrix with values ranging from 0 to 8

# In[17]:


np.arange(0,9).reshape(3,3)


# #### Create a 3x3 identity matrix

# In[18]:


np.eye(3)


# #### Use NumPy to generate a random number between 0 and 1

# In[73]:


np.random.rand()


# #### Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution

# In[26]:


np.random.randn(5,5)


# #### Create the following matrix:

# In[76]:


np.arange(1,101).reshape(10,10)/100


# #### Create an array of 20 linearly spaced points between 0 and 1:

# In[49]:


np.linspace(0,1,20)


# ## Numpy Indexing and Selection
# 
# Now you will be given a few matrices, and be asked to replicate the resulting matrix outputs:

# In[51]:


mat = np.arange(1,26).reshape(5,5)
mat


# In[52]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
mat[2:, 1:]


# In[40]:





# In[55]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
mat[3, 4]


# In[41]:





# In[59]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
mat[:3, 1].reshape((3,1))


# In[42]:





# In[31]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[63]:


mat[4]


# In[64]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
mat[3:]


# In[49]:





# ### Now do the following

# #### Get the sum of all the values in mat

# In[65]:


np.sum(mat)


# #### Get the standard deviation of the values in mat

# In[68]:


np.std(mat)


# #### Get the sum of all the columns in mat

# In[78]:


mat.sum(axis=0)  


# # Great Job!
