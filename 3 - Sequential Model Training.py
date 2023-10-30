#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import tensorflow as tf


# In[2]:


#pip install opencv-python


# In[3]:


#pip install tensorflow


# In[4]:


df=pd.read_csv('labels.csv')


# In[5]:


df.head()


# In[6]:


import xml.etree.ElementTree as xet


# In[7]:


filename=df['filepath'][0]
filename


# In[8]:


def getFilename(filename):
    filename_image=xet.parse(filename).getroot().find('filename').text
    filepath_image=os.path.join('./images',filename_image)
    return filepath_image


# In[9]:


getFilename(filename)


# In[10]:


image_path=list(df['filepath'].apply(getFilename))
image_path


# In[11]:


file_path = image_path[0]
file_path


# In[12]:


img = cv2.imread(file_path)

cv2.namedWindow('example', cv2.WINDOW_NORMAL)
cv2.imshow('example',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


cv2.rectangle(img,(1093,645),(1396,727),(0,255,0),3)
cv2.namedWindow('example', cv2.WINDOW_NORMAL)
cv2.imshow('example',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[14]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img,img_to_array


# In[15]:


labels=df.iloc[:,1:].values


# In[16]:


data=[]
output=[]

for ind in range(len(image_path)):
    image=image_path[ind]
    img_arr=cv2.imread(image)
    h,w,d=img_arr.shape
    #preprocessing
    load_image=load_img(image,target_size=(224,224))
    load_image_arr=img_to_array(load_image)
    norm_load_image_arr=load_image_arr/255.0  #normalization
    #normalization to labels
    xmin,xmax,ymin,ymax=labels[ind]
    nxmin,nxmax=xmin/w,xmax/w
    nymin,nymax=ymin/h,ymax/h
    label_norm=(nxmin,nxmax,nymin,nymax)  #normalized output
    #------------------Append------
    data.append(norm_load_image_arr)
    output.append(label_norm)


# In[17]:


X =np.array(data,dtype='float32')
y=np.array(output,dtype='float32')


# In[18]:


X.shape, y.shape


# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ### Sequential Model

# In[20]:


i=tf.keras.layers.Input(shape=(224,224,3))
x=tf.keras.layers.Conv2D(64,(5,5),activation='relu')(i)
x=tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(128, (5,5), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(256, (7,7), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(512, (7,7), activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256,activation='relu')(x)
x = tf.keras.layers.Dense(128,activation='relu')(x)
x = tf.keras.layers.Dense(64,activation='relu')(x)
o = tf.keras.layers.Dense(4,activation='sigmoid')(x)


model = tf.keras.Model(inputs=[i], outputs=[o])


# In[21]:


# Compile the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))


# In[22]:


# Print the model summary
model.summary()


# ### Model Training

# In[23]:


from tensorflow.keras.callbacks import TensorBoard


# In[24]:


# Create a TensorBoard callback for visualization
tfb = TensorBoard('object_detection_Sequential')


# In[25]:


# Train the model
history = model.fit(x=X_train, y=y_train, batch_size=10, epochs=100, validation_data=(X_test, y_test), callbacks=[tfb])


# In[29]:


model.save('./models/object_detection_sequential.h5')


# In[ ]:




