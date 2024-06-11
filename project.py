#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data collection and anality
# PIma diabetes dataset
# 

# In[20]:


# landing the diabetes data set
diabetes_dataset = pd.read_csv('diabetes.csv')


# pd.read_csv?

# In[21]:


#printing time 5 rows
diabetes_dataset.head()


# In[22]:


diabetes_dataset.shape


# In[23]:


#getting the statisitial
diabetes_dataset.describe()


# In[24]:


diabetes_dataset['Outcome'].value_counts()


# 0 = tidak diabetes
# 1 = diabetes

# In[25]:


diabetes_dataset.groupby('Outcome').mean()


# In[26]:


X = diabetes_dataset.drop(columns = 'Outcome', axis =1)
Y = diabetes_dataset['Outcome']


# In[27]:


print(X)


# In[28]:


print(Y)


# data stadarisasi

# In[29]:


scaler = StandardScaler()


# In[30]:


scaler.fit(X)


# In[31]:


standardized_data = scaler.transform(X)


# In[32]:


print(standardized_data)


# In[33]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[34]:


print(X)
print(Y)


# train test split

# In[35]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[36]:


print(X.shape, X_train.shape, X_test.shape)


# In[37]:


classifier = svm.SVC (kernel='linear')


# In[38]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[39]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy =  accuracy_score(X_train_prediction, Y_train)


# In[40]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[42]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[43]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[45]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

