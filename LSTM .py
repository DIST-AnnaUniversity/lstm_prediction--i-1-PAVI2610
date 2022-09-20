#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df_data=pd.DataFrame()
x1=[]
x2=[]
x3=[]
y=[]
for i in range(1,1001):
    x1=x1+[i]
    x2=x2+[i+1]
    x3=x3+[i+2]
    y=y+[i+3]
df_data['x1']=x1
df_data['x2']=x2
df_data['x3']=x3
df_data['y']=y
df_data.to_csv("data.csv")


# In[2]:


import tensorflow
import numpy as np
#tensorflow.VERSION
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed


# In[3]:


import numpy as np
x=np.array(df_data.iloc[:,0:3])
y=np.array(df_data.iloc[:,3])
x,y


# In[4]:


x=x.reshape(1000,3,1)
x.shape


# In[5]:


y=y.reshape(1000,1,1)
y.shape


# In[6]:


model=Sequential()
model.add(LSTM(input_shape=(3,1),
                    units=3,
                    activation='relu'))
model.add(Dense(activation='linear', units=1))
model.compile(loss = 'mse', optimizer = 'rmsprop')
np.random.seed(7)
model.fit(x, y, epochs = 500, batch_size = 8)


# In[7]:


t1 = np.array([1996,1997,1998])
t1.shape


# In[8]:


t1= t1.reshape(1,3,1)
model.predict(t1)


# In[ ]:




