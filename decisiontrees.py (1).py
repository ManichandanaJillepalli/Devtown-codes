#!/usr/bin/env python
# coding: utf-8

# ## Import Necessary Libraries

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split
import seaborn as sns


# ## Data Collection

# In[35]:


com_data=pd.read_csv('Company_Data.csv',sep=',')
com_data


# ## Data Understanding
#    Initial Analysis

# In[36]:


com_data.shape


# In[37]:


com_data.isna().sum()


# In[38]:


com_data.info()


# In[39]:


com_data.head()


# In[40]:


com_data.describe().T


# In[41]:


com_data.dtypes


# ## Outliers Check

# In[42]:


import warnings
warnings.filterwarnings('ignore')


# In[43]:


data1=sns.boxplot(com_data['Sales'])


# ## The data has 2 outliers

# In[44]:


plt.rcParams['figure.figsize']=9,4


# In[45]:


plt.figure(figsize=(15,5))
print("Skew: {}".format(com_data['Sales'].skew()))
print("Kurtosis: {}".format(com_data['Sales'].kurtosis()))
data1 = sns.kdeplot(com_data['Sales'],shade=True,color='g')
plt.xticks([i for i in range(0,20,1)])
plt.show()


# In[46]:


obj_colum = com_data.select_dtypes(include='object').columns.tolist()


# In[47]:


plt.figure(figsize=(16,10))
for i,col in enumerate(obj_colum,1):
    plt.subplot(2,2,i)
    sns.countplot(data=com_data,y=col)
    plt.subplot(2,2,i+1)
    com_data[col].value_counts(normalize=True).plot.bar()
    plt.ylabel(col)
    plt.xlabel('% distribution per category')
plt.tight_layout()
plt.show()  


# In[48]:


num_columns = com_data.select_dtypes(exclude='object').columns.tolist()


# In[49]:


plt.figure(figsize=(16,30))
for i,col in enumerate(num_columns,1):
    plt.subplot(8,4,i)
    sns.kdeplot(com_data[col],color='g',shade=True)
    plt.subplot(8,4,i+10)
    com_data[col].plot.box()
plt.tight_layout() 
plt.show()
num_data = com_data[num_columns]
pd.DataFrame(data=[num_data.skew(),num_data.kurtosis()],index=['skewness','kurtosis'])


# In[50]:


corr=com_data.corr()


# In[51]:


com_data = pd.get_dummies(com_data,columns = ['ShelveLoc', 'Urban', 'US'])


# In[52]:


corr=com_data.corr()


# In[53]:


plt.figure(figsize=(15,15))
sns.heatmap(corr,annot=True)


# ## Decision tree--Model

# In[54]:


com_data["sales"]="small"
com_data.loc[com_data["Sales"]>7.49,"sales"]="large"
com_data.drop(["Sales"],axis=1,inplace=True)


# In[55]:


X = com_data.iloc[:,0:14]
y = com_data.iloc[:,14]


# In[56]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, stratify = y)


# In[57]:


y_train.value_counts()


# In[58]:


model = DT(criterion='entropy') 
model.fit(x_train,y_train)


# In[59]:


pred_train = model.predict(x_train)
#accuracy_score(y_train,pred_train)


# In[60]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train,pred_train)


# In[61]:


pred_test = model.predict(x_test)
#accuracy_score(y_test,pred_test)


# In[62]:


confusion_matrix(y_test,pred_test)


# In[63]:


com_data_t=pd.DataFrame({'Actual':y_test, 'Predicted':pred_test})


# In[64]:


com_data_t


# In[ ]:




