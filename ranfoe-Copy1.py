#!/usr/bin/env python
# coding: utf-8

# ## Import Necessary Libraries

# In[49]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# ## Data collection

# In[4]:


df = pd.read_csv('Fraud_check (1).csv')


# In[5]:


fraud=df.copy()


# In[6]:


fraud.head()


# In[7]:


fraud.describe().T


# In[8]:


fraud.isnull().sum()


# In[9]:


fraud.dtypes


# In[10]:


import warnings
warnings.filterwarnings('ignore')


# ## Outlier Check

# In[11]:


ax = sns.boxplot(fraud['Taxable.Income'])


# ## There are no outliers in the data

# In[12]:


plt.rcParams["figure.figsize"] = 9,5


# In[13]:


plt.figure(figsize=(16,5))
print("Skew: {}".format(fraud['Taxable.Income'].skew()))
print("Kurtosis: {}".format(fraud['Taxable.Income'].kurtosis()))
ax = sns.kdeplot(fraud['Taxable.Income'],shade=True,color='g')
plt.xticks([i for i in range(10000,100000,10000)])
plt.show()


# In[14]:


obj_colum = fraud.select_dtypes(include='object').columns.tolist()


# In[15]:


plt.figure(figsize=(16,10))
for i,col in enumerate(obj_colum,1):
    plt.subplot(2,2,i)
    sns.countplot(data=fraud,y=col)
    plt.subplot(2,2,i+1)
    fraud[col].value_counts(normalize=True).plot.bar()
    plt.ylabel(col)
    plt.xlabel('% distribution per category')
plt.tight_layout()
plt.show() 


# In[16]:


num_columns = fraud.select_dtypes(exclude='object').columns.tolist()


# In[17]:


plt.figure(figsize=(18,40))
for i,col in enumerate(num_columns,1):
    plt.subplot(8,4,i)
    sns.kdeplot(df[col],color='g',shade=True)
    plt.subplot(8,4,i+10)
    df[col].plot.box()
plt.tight_layout() 
plt.show()
num_data = df[num_columns]
pd.DataFrame(data=[num_data.skew(),num_data.kurtosis()],index=['skewness','kurtosis'])


# In[18]:


fraud = pd.get_dummies(fraud, columns = ['Undergrad','Marital.Status','Urban'])


# In[19]:


corr = fraud.corr()


# In[20]:


plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True)


# ## Randome Forest Model

# In[21]:


fraud['Taxable.Income']=pd.cut(fraud['Taxable.Income'],bins=[0,30000,100000],labels=['risky','good'])


# In[22]:


list(fraud.columns)


# In[23]:


X = fraud.iloc[:,1:10]
y = fraud.iloc[:,0]


# In[24]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[25]:


y_train.value_counts()


# In[26]:


model =RF(n_jobs=4,n_estimators = 150, oob_score =True,criterion ='entropy') 
model.fit(x_train,y_train)
model.oob_score_


# In[27]:


pred_train = model.predict(x_train)


# In[28]:


accuracy_score(y_train,pred_train)


# In[29]:


confusion_matrix(y_train,pred_train)


# In[30]:


pred_test = model.predict(x_test)


# In[31]:


accuracy_score(y_test,pred_test)


# In[32]:


confusion_matrix(y_test,pred_test)


# In[33]:


df_t=pd.DataFrame({'Actual':y_test, 'Predicted':pred_test})


# In[34]:


df_t

