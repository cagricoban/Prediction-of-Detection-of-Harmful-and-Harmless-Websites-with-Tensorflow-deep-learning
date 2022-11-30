#!/usr/bin/env python
# coding: utf-8

# # Prediction of Detection of Harmful and Harmless Websites with Tensorflow deep learning.

# Analysis study on the detection of harmful and harmless websites. A value of "1" in the Type variable indicates that the website is infected, and the variable "0" indicates that the website is harmless. Our goal here is to predict whether the site is malicious or harmless by building a model.

# We install the necessary libraries.

# In[1]:


import pandas as pd
import numpy as np


# We load our Dataset.

# In[3]:


dataFrame= pd.read_excel("maliciousornot.xlsx")


# In[4]:


dataFrame


# In[5]:


dataFrame.info()


# When we examine our variables in the data set; We can see that all except Type are floats, that is, numeric variables.

# In[7]:


dataFrame.describe()


# When we look at the summary of the data, especially the "Type" average is 0.38 for "1". That says this: We conclude that there are more harmless sites than malicious sites in our dataset.

# In[9]:


dataFrame.corr()["Type"].sort_values()


# When we look at the correlations of the data according to the types, the probability of the "type" being "1" increases as the "source_k" variable increases. The positive correlation between these two variables is 78%. The variable "URL_length" if the variable "Type" has the highest negative correlation relationship. There is a -22% negative correlation between these two variables. So the higher the "URL_length", the more likely the type is to be "0".

# ## Data visualization

# In[11]:


import matplotlib.pyplot as plt
import seaborn as sbn


# In[12]:


sbn.countplot(x="Type", data=dataFrame)


# When we visualize the types, when we look at the line numbers of the 0 and 1 types, there is no difference that will hinder the training and deviate from the estimation. Type distributions are appropriately distributed so that the model can be built from the data and trained.

# In[13]:


dataFrame.corr()["Type"].sort_values().plot(kind="bar")


# The relationship between the variables is given in the bar graph.

# ## Model building

# The dependent variable (y) of the model is parsed as type and the independent variables as arrays.

# In[14]:


y=dataFrame["Type"].values
x= dataFrame.drop("Type",axis=1).values


# The data set is divided into two as training and testing. 70% of the data set is reserved for training and 30% for testing. The dataset is divided into training and testing using the sklearn library.

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3,random_state=15 )


# ### Normalization Process

# We apply Max-Min Normalization to scale the data to a scale of 0-1. Normalization processing units will ensure that many different data are not mixed with each other. We do this with the MinMaxScaler class in the sklearn library.

# In[17]:


from sklearn.preprocessing import MinMaxScaler


# In[18]:


scaler = MinMaxScaler()


# In[19]:


scaler.fit(x_train)


# In[20]:


x_train = scaler.transform(x_train)


# In[21]:


x_test = scaler.transform(x_test)


# ### Tensorflow

# In[51]:


import tensorflow as  tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[27]:


x_train.shape


# Our deep network consists of 4 layers. The input layer is the two middleware and the output layer. We have 30 neurons in our input layer, 15 neurons in our intermediate layers and 1 neuron in our output layer. The reason we chose 30 neurons in our input layer is because we have 30 independent input variables. The reason why the intermediate layers are given as 15 is generally stated in the literature that the number of neurons in the intermediate layers should be between the number of neurons in the input and output layers.The activation function of the neurons in the input and intermediate layers was determined as "relu", and the activation function of the neurons in the output layer was determined as "sigmoid".The loss function is "binary_crossentropy" and the optimization function is "man". The main reason we use the "binary_crossentropy" of the missing function is because the model is a classification model.

# In[29]:


model = Sequential()

model.add(Dense(units=30, activation= "relu"))
model.add(Dense(units=15, activation= "relu"))
model.add(Dense(units=15, activation= "relu"))
model.add(Dense(units=1, activation= "sigmoid"))

model.compile(loss="binary_crossentropy",optimizer= "adam")


# It was formed as a result of 700 iterations given the training and test data with the model.

# In[31]:


model.fit(x=x_train, y=y_train, epochs=700, validation_data =(x_test,y_test),verbose=1)


# In[33]:


modelKaybi = pd.DataFrame(model.history.history)


# In[34]:


modelKaybi.plot()


# "loss" and "validation loss" decrease up to the 70th iteration. but after the 70th iteration, "loss" decreases while "validation loss" increases. An overfitting event occurred as a result of our training. As a result, the test data went to model memorization. When a new value comes, he will not be able to make a correct guess. He has worked hard in himself while trying to be consistent.

# ### Early Stopping

# In[35]:


model = Sequential()

model.add(Dense(units=30, activation= "relu"))
model.add(Dense(units=15, activation= "relu"))
model.add(Dense(units=15, activation= "relu"))
model.add(Dense(units=1, activation= "sigmoid"))

model.compile(loss="binary_crossentropy",optimizer= "adam")


# We encountered an overfitting problem in our model. To avoid this, we must stop early. The early stop function will allow us to stop our iteration while our training and test losses are at a minimum, so we can prevent overfitting.We choose "monitor", which determines the metric we will verify, because our most important parameter here is "valuation loss" as our verification metric. "Patience" looks at iterations. For example, we chose 25, if there is no improvement after 25 iterations, it will stop. "Mode" allows the metric we choose to stop at whatever level we want it to be.

# In[38]:


earlyStopping = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)


# In[39]:


model.fit(x=x_train, y=y_train ,epochs=700 ,validation_data=(x_test,y_test), verbose= 1, callbacks=[earlyStopping])


# After early stopping, our training stopped at the 114th iteration.

# In[40]:


modelKaybi = pd.DataFrame(model.history.history)


# In[41]:


modelKaybi.plot()


# When we plotted our losses, there was a serious improvement.

# ## Dropout

# In[42]:


model = Sequential()

model.add(Dense(units=30, activation= "relu"))
model.add(Dropout(0.5))

model.add(Dense(units=15, activation= "relu"))
model.add(Dropout(0.5))

model.add(Dense(units=15, activation= "relu"))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation= "sigmoid"))

model.compile(loss="binary_crossentropy",optimizer= "adam")


# The dropout function is applied again when we encounter an overfitting situation in our model. The dropout function randomly removes neurons in layers. The model was rebuilt and the Droput function was added after each layer. Dropout 0.5 and below is the appropriate method. We have given the droput function 0.5 in our model. We redesigned our network structure.

# In[43]:


model.fit(x=x_train, y=y_train ,epochs=700 ,validation_data=(x_test,y_test), verbose= 1, callbacks=[earlyStopping])


# In[44]:


kayipDf = pd.DataFrame(model.history.history)


# In[46]:


kayipDf.plot()


# After applying the dropout function, we examine our predictions.

# In[54]:


predictions= (model.predict(x_test) > 0.5).astype("int32")


# In[55]:


predictions


# ### Report

# We get the report of the outputs in the classification.

# In[56]:


from sklearn.metrics import classification_report, confusion_matrix


# In[58]:


print(classification_report(y_test,predictions))


# Result: Looking at the output, zeros were predicted with 89% accuracy and ones with 91% accuracy.

# In[ ]:




