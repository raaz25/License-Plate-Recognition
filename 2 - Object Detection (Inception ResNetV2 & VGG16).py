#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2


# In[2]:


df=pd.read_csv('labels.csv')


# In[3]:


df.head()


# In[4]:


import xml.etree.ElementTree as xet


# In[5]:


filename=df['filepath'][0]
filename


# In[6]:


def getFilename(filename):
    filename_image=xet.parse(filename).getroot().find('filename').text      # to extract the filename from the XML
    filepath_image=os.path.join('./images',filename_image)                  # the path is constructed correctly (xml to jpeg)
    return filepath_image


# In[7]:


getFilename(filename)


# In[8]:


image_path=list(df['filepath'].apply(getFilename))
image_path


# In[9]:


s='./images\\N31.jpeg'
if s in image_path:
    index = image_path.index(s)
    print(f"The element {s} is at index {index}.")
else:
    print(f"The element {s} is not in the list.")


# ### Verify Image and Output

# In[10]:


file_path = image_path[158]
file_path


# In[11]:


img = cv2.imread(file_path)

cv2.namedWindow('example', cv2.WINDOW_NORMAL)
cv2.imshow('example',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[12]:


cv2.rectangle(img,(256,233),(381,261),(0,255,0),3)  #(1093,645),(1396,727),(0,255,0) (xmin, ymin, xmax, ymax)
cv2.namedWindow('example', cv2.WINDOW_NORMAL)
cv2.imshow('example',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Data Preprocessing

# In[13]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img,img_to_array


# In[14]:


labels=df.iloc[:,1:].values


# In[15]:


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


# In[16]:


X = np.array(data,dtype='float32')
y = np.array(output,dtype='float32')


# In[17]:


X.shape, y.shape


# In[18]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# # Deep Learning Model

# ### InceptionResNetV2

# In[19]:


from keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
import tensorflow as tf


# In[20]:


inception_resnet = InceptionResNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))
inception_resnet.trainable=False
#------------------------
headmodel=inception_resnet.output
headmodel=Flatten()(headmodel)
headmodel=Dense(500,activation='relu')(headmodel)
headmodel=Dense(250,activation='relu')(headmodel)
headmodel=Dense(4,activation='sigmoid')(headmodel)
#-----------
model=Model(inputs=inception_resnet.input,outputs=headmodel)


# In[21]:


#pip install --upgrade tensorflow


# In[22]:


#pip install --upgrade keras


# In[33]:


# compile 
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.summary()


# # Model Training

# In[34]:


from tensorflow.keras.callbacks import TensorBoard


# In[35]:


tfb=TensorBoard('object_detection')


# In[36]:


history=model.fit(x=X_train, y=y_train, batch_size=10,epochs=100,validation_data=(X_test,y_test),callbacks=[tfb])


# In[27]:


history=model.fit(x=X_train, y=y_train, batch_size=10,epochs=200,validation_data=(X_test,y_test),callbacks=[tfb],initial_epoch=101)


# In[28]:


model.save('./models/object_detection.h5')


# ### VGG16

# In[37]:


from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


# In[38]:


# Load the VGG16 model with pre-trained weights
vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Freeze the layers of the pre-trained VGG16 model
for layer in vgg16.layers:
    layer.trainable = False

# Add your custom head on top of VGG16
headmodel = vgg16.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500, activation='relu')(headmodel)
headmodel = Dense(250, activation='relu')(headmodel)
headmodel = Dense(4, activation='sigmoid')(headmodel)

# Create the complete model
model = Model(inputs=vgg16.input, outputs=headmodel)


# In[39]:


# Compile the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))


# In[40]:


# Print the model summary
model.summary()


# ## Model Training

# In[41]:


from tensorflow.keras.callbacks import TensorBoard


# In[42]:


# Create a TensorBoard callback for visualization
tfb = TensorBoard('object_detection_VGG16')


# In[43]:


# Train the model
history = model.fit(x=X_train, y=y_train, batch_size=10, epochs=100, validation_data=(X_test, y_test), callbacks=[tfb])


# In[44]:


model.save('./models/object_detection_VGG16.h5')


# In[ ]:




