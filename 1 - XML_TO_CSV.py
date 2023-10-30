#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import xml.etree.ElementTree as xet 


# In[5]:


from glob import glob   #to extract the file path


# In[6]:


path = glob('./image/*.xml')
path


# In[7]:


labels_dict=dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])

for filename in path:
    #filename = path[0]
    info=xet.parse(filename)
    root=info.getroot()
    member_object=root.find('object')
    labels_info=member_object.find('bndbox')   #bndbox - bounding box
    xmin=int(labels_info.find('xmin').text)
    xmax=int(labels_info.find('xmax').text)
    ymin=int(labels_info.find('ymin').text)
    ymax=int(labels_info.find('ymax').text)

    #print(xmin,xmax,ymin,ymax)
    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)


# In[8]:


labels_dict


# In[9]:


df=pd.DataFrame(labels_dict)
df


# In[11]:


df.to_csv('labels.csv',index=False)


# In[ ]:




