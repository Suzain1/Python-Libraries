#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


titanic = sns.load_dataset('titanic')
sns.barplot(x='embark_town', y= 'age', data = titanic, palette='PuRd', ci =None)


# In[11]:


sns.barplot(x='age', y= 'embark_town', data = titanic, palette='PuRd', ci =None, orient='h')


# In[13]:


df = sns.load_dataset('iris')
sns.distplot(df['petal_length'],kde = False)


# In[14]:


titanic = sns.load_dataset('titanic')
sns.countplot(x = 'class', hue = 'who', data = titanic, palette = 'magma')


# In[ ]:




