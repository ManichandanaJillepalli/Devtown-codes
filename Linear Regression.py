#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries

# In[9]:


import pandas as pd
import seaborn as sns


# ## Import data

# In[4]:


paper_data = pd.read_csv('NewspaperData.csv')
paper_data.head()


# In[5]:


paper_data.shape 


# In[6]:


paper_data.isna().sum() # if there are null values Nan


# In[8]:


paper_data.describe()


# In[10]:


sns.regplot(x = 'daily', y='sunday', data = paper_data )


# In[12]:


import statsmodels.formula.api as smf


# In[13]:


linear_model = smf.ols(formula = 'sunday~daily', data = paper_data).fit()        ## ordinary least square                      


# In[14]:


linear_model.params


# In[18]:


prediction_data = {'daily':[600, 1000, 1500]}
prediction_data


# In[19]:


test_data = pd.DataFrame(data=prediction_data)
test_data


# In[20]:


linear_model.predict(test_data)


# In[ ]:




