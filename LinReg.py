
# coding: utf-8

# In[1]:


import os
os.chdir("C:/Users/Admin/Desktop/ml")


# In[5]:


import pandas as pd
import numpy as np


# In[6]:


ls


# In[7]:


df = pd.read_csv("train (1).csv")


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


correlation_values = df.select_dtypes(include=(np.number)).corr()


# In[11]:


correlation_values


# In[25]:


(correlation_values["SalePrice"] >= 0.6) | (correlation_values["SalePrice"] <=-0.6)


# In[50]:


x = df[["OverallQual","TotalBsmtSF","GrLivArea","GarageArea","GarageCars","1stFlrSF"]]


# In[51]:


y = df["SalePrice"]


# In[52]:


from sklearn.model_selection import train_test_split as tts


# In[53]:


x_train,x_test,y_train,y_test = tts(x,y,test_size=0.3,random_state=42)


# In[54]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()


# In[55]:


reg.fit(x_train,y_train)


# In[56]:


y_pred = reg.predict(x_test)


# In[57]:


reg.score(x_test,y_test)


# In[58]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test,y_pred))


# In[59]:


rmse

