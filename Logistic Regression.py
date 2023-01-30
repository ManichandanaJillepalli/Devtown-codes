#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[7]:


import pandas as pd 


# In[9]:


logreg_data=pd.read_csv('logreg.csv')
logreg_data


# In[10]:


logreg_data.shape


# In[11]:


logreg_data.isna().sum()


# In[12]:


logreg_data.head(20)


# In[23]:


logreg_data.dropna(axis = 0, inplace = True)


# In[24]:


logreg_data.head(20)


# In[25]:


logreg_data.isna().sum()


# ## seperating input and output features

# In[28]:


X = logreg_data.drop(labels='ATTORNEY', axis=1)
y = logreg_data[['ATTORNEY']]


# In[29]:


X


# In[30]:


y


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=12)


# In[37]:


X_train


# In[38]:


X_test


# In[39]:


y_train


# In[40]:


y_test


# In[42]:


X_train.shape, y_train.shape
X_test.shape, y_test.shape


# In[43]:


import warnings                       ## Model training
warnings.filterwarnings('ignore')


# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
## Logistic  regression 

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


# In[51]:


from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
plt.figure(figsize =( 16, 10))


# In[52]:


plot_tree(dt_model, rounded= True, filled= True)
plt.show()  


# In[53]:


y_pred_train = dt_model.predict(X_train)
y_pred_train


# In[56]:


from sklearn.metrics import accuracy_score


# In[57]:


accuracy_score(y_train, y_pred_train)


# In[59]:


y_test_pred = dt_model.predict(X_test)
y_test_pred


# In[60]:


accuracy_score(y_test, y_test_pred)


# In[ ]:




