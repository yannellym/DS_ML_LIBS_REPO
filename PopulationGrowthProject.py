#!/usr/bin/env python
# coding: utf-8

# In[28]:


# import libraries to use
import pandas as pd
import numpy as np


# In[29]:


#check the directory
get_ipython().system('pwd')


# In[30]:


# list the items in the directory to make sure data is there
ls


# In[52]:


# load the data set
data = pd.read_csv('world_pop_growth.csv')
# check the head of the data
data.head()


# In[54]:


# .info() to find out more information about the data
data.info()
# 266 entries
# 66 columns


# In[71]:


# look for null values to either replace them or drop them
data.isnull().sum()
# looks like the 1960 column has a lot of null values (266)


# In[85]:


# drop the column 1960 since it had a lot of null values
data = data.drop(['1960'], axis = 1)
data


# In[111]:


# dropping the unecessary columns from our data
data = data.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],axis=1)
data


# In[112]:


import seaborn as sns


# In[174]:


column_sums = data.sum()
# create a DataFrame with two columns: 'years' and 'growth'
df = pd.DataFrame({
    'years': column_sums.index.astype(int),
    'growth': column_sums.values
})
df


# In[175]:


import matplotlib.pyplot as plt


# In[301]:


# Creating a scatter plot
plt.scatter(df['years'], df['growth'])
plt.xlabel('Years')
plt.ylabel('Growth')
plt.show()
# there is a general trend of decreasing growth over time.


# In[302]:


# plot the data as a line graph
fig, gr = plt.subplots(figsize=(12, 6))  # set the figure size
# plot the data as a line graph
df.plot(x='years', y='growth', kind='line', color='red',linestyle='-.',linewidth=2, ax= gr)


# In[192]:


# add axis labels and a title
gr.set_xlabel('Year')
gr.set_ylabel('Population Growth (billions)')
gr.set_title('World Population Growth')
fig


# In[198]:


# set the intervals to be by 5
gr.set_xticks(df['years'][::5])
# # display the plot
plt.show()
fig


# In[207]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# # our model
# 
# 

# In[304]:


X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[294]:





# In[305]:


# linear regression model since we are going to be predicting continuous values
model = LinearRegression()


# In[306]:


# fit the data to our model
model.fit(X_train, y_train)


# In[308]:


# Make predictions on the testing data
y_pred = model.predict(X_test)


# In[314]:


# Predicting the growth for future years
future_years = [[2022], [2023], [2024], [2025], [2026], [2027], [2028], [2029], [2030]]
future_growth = model.predict(future_years)
future_growth


# In[317]:





# In[ ]:





# In[ ]:




