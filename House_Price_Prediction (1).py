#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Import Libraries
import pandas as pd 
import seaborn as sns


# In[20]:


# Load the dataset
house_details_data = pd.read_csv("C:/Users/ACER/Downloads/HousePricePrediction (3).csv")


# In[22]:


print(house_details_data)


# In[30]:


# Data Preprocessing
house_details_data = pd.get_dummies(house_details_data, drop_first=True)
house_details_data = house_details_data.dropna()
print(house_details_data)


# In[31]:


sns.scatterplot(x=house_details_data["YearBuilt"],y=house_details_data["SalePrice"],hue = house_details_data["LotArea"])


# In[34]:


# Define Features (X) and Target (y)
X = house_details_data.drop(columns = ["SalePrice"])
y = house_details_data["SalePrice"]


# In[38]:


# Split the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# In[54]:


# Train Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[55]:


# Predict Values
y_pred = model.predict(X_test)


# In[58]:


y_pred


# In[59]:


y_test


# In[60]:


# Evaluate 
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print("r-squared:", r2)
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1-r2)*(n-1)/(n-p-1))
print("Adjusted r^2:", adjusted_r2)


# In[61]:


X_test.shape


# In[ ]:





# In[ ]:





# In[ ]:




