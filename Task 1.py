#!/usr/bin/env python
# coding: utf-8

# # prediction using supervised ML 

# Author - Snehal Sawant

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


data=pd.read_csv('student_scores - student_scores.csv')
data


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.plot(kind='scatter',x='Hours',y='Scores')
plt.show


# In[8]:


data.corr(method='pearson')


# In[9]:


data.corr(method='spearman')


# In[10]:


hours=data['Hours']
scores=data['Scores']


# In[11]:


sns.distplot(hours)


# In[13]:


sns.distplot(scores)


# In[14]:


x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[15]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=50)


# In[16]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train, y_train)


# In[17]:


m=reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[18]:


y_pred=reg.predict(x_test)


# In[19]:


actual_predicted=pd.DataFrame({'Traget':y_test,'Predicted':y_pred})
actual_predicted


# In[20]:


sns.set_style('white')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# what would be the predicted score if a student studies for 9.25 hours/days

# In[26]:


h=9.25
s=reg.predict([[h]])
print('If a student studies for {} hours per days he will score {} % in exam'.format(h,s))


# In[ ]:




