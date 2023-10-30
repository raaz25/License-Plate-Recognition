#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


from tensorflow.keras.preprocessing.image import load_img,img_to_array


# In[7]:


#load model
model = tf.keras.models.load_model('./models/object_detection_sequential.h5')
print('Model loaded successfully')


# In[37]:


# Create Pipeline
path = './test_image/26.jpeg'
def object_detection(path):
    #read image
    image=load_img(path) #PIL object
    image = np.array(image,dtype=np.uint8) #8 bit array (0,255)
    image1=load_img(path,target_size=(224,224))
    #data preprocessing
    image_arr_224=img_to_array(image1)/255.0   #convert into array and get the normalized output
    # size of the original image
    h,w,d = image.shape
    test_arr=image_arr_224.reshape(1,224,224,3)
    # Make Predictions
    coord=model.predict(test_arr)
    # denormalize the values
    denorm = np.array([w,w,h,h])
    coords=coord*denorm
    coords=coords.astype(np.int32)
    # draw bounding on the top of image
    xmin,xmax,ymin,ymax=coords[0]
    pt1=(xmin,ymin)
    pt2=(xmax,ymax)
    pt1,pt2
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    return image, coords


# In[38]:


path = './test_image/26.jpeg'
image, cods = object_detection(path)
print(cods)

plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()


# ### Optical Character Recognition

# In[33]:


#pip install pytesseract


# In[57]:


import pytesseract as pt


# In[58]:


path = './test_image/26.jpeg'
image, cods = object_detection(path)
print(cods)

plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()


# In[59]:


img =np.array(load_img(path))
xmin,xmax,ymin,ymax=cods[0]
roi=img[ymin:ymax,xmin:xmax]


# In[60]:


plt.imshow(roi)
plt.show()


# In[65]:


pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# In[66]:


text=pt.image_to_string(roi)
print(text)

