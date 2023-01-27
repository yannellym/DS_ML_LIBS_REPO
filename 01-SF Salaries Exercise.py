#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../../Pierian_Data_Logo.png' /></a>
# ___

# # SF Salaries Exercise 
# 
# Welcome to a quick exercise for you to practice your pandas skills! We will be using the [SF Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries) from Kaggle! Just follow along and complete the tasks outlined in bold below. The tasks will get harder and harder as you go along.

# ** Import pandas as pd.**

# In[2]:


import pandas as pd


# ** Read Salaries.csv as a dataframe called sal.**

# In[117]:


sal = pd.read_csv('Salaries.csv')
sal.head(2)


# ** Check the head of the DataFrame. **

# In[9]:


sal.head()


# ** Use the .info() method to find out how many entries there are.**

# In[10]:


sal.info()


# **What is the average BasePay ?**

# In[14]:


sal['BasePay'].mean()


# ** What is the highest amount of OvertimePay in the dataset ? **

# In[15]:


sal['OvertimePay'].max()


# ** What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll). **

# In[46]:


sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']


# ** How much does JOSEPH DRISCOLL make (including benefits)? **

# In[47]:


sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']


# ** What is the name of highest paid person (including benefits)?**

# In[118]:


max_sal = sal['TotalPayBenefits'].max()
sal[sal['TotalPayBenefits'] == max_sal]['EmployeeName']


# ** What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?**

# In[119]:


min_sal = sal['TotalPayBenefits'].min()
sal[sal['TotalPayBenefits'] == min_sal]['EmployeeName']

# Joe Lopez seems to have no base pay or benefits. He only has 'OtherPay', which seems to be a negative amount.


# ** What was the average (mean) BasePay of all employees per year? (2011-2014) ? **

# In[78]:


sal.pivot_table(values='BasePay', index="Year")


# ** How many unique job titles are there? **

# In[85]:


sal['JobTitle'].nunique()


# ** What are the top 5 most common jobs? **

# In[120]:


sal['JobTitle'].value_counts().head(5)


# ** How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?) **

# In[121]:


sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1)


# ** How many people have the word Chief in their job title? (This is pretty tricky) **

# In[129]:


def chief_string(title):
    if "chief"in title.lower():
        return True
    else:
        return False


# In[130]:


sum(sal['JobTitle'].apply(lambda x: chief_string(x)))


# ** Bonus: Is there a correlation between length of the Job Title string and Salary? **

# In[131]:


sal['title_len'] = sal['JobTitle'].apply(len)


# In[133]:


sal[['TotalPayBenefits', 'title_len']].corr()

# no correlation between how long the title is and how much you get paid


# # Great Job!
