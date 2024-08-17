#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


movie_info = pd.read_csv('C:\\Users\\suzai\\Downloads\\archive (13)\\NetflixOriginals.csv', encoding = 'latin')


# In[6]:


movie_info.head(5)


# In[8]:


movie_info.shape


# In[ ]:


#SPLIT


# In[9]:


#Grouping based on Genre
group = movie_info.groupby('Genre')


# In[13]:


group.groups


# In[19]:


len(group)


# In[21]:


group.size()


# In[25]:


#grouping based on genre and language
group2 = movie_info.groupby(['Genre','Language'])


# In[30]:


group2.size()


# In[32]:


#Forming subset of data
group.get_group('Comedy')


# In[35]:


#iterating throug groups
for name, groups in group:
    print("Group Name", name)
    print(groups[['Title','Genre']],'\n')
          


# In[36]:


for name, groups in group2:
    print("Group Name:", name)
    print(groups,'\n')
    print('______________________________End of␣→group_______________________________')


# In[37]:


group2.first()


# In[39]:


group2.last()


# In[41]:


group3 = movie_info.groupby(['Genre'], sort = False)


# In[43]:


group3.groups


# In[ ]:


#APPLY


# In[45]:


movie_info.groupby('Genre')[['Runtime']].mean()


# In[46]:


movie_info.groupby(['Genre','Language']).size()


# In[49]:


movie_info.groupby(['Genre','IMDB Score']).min()


# In[50]:


movie_info.groupby(['Genre'])['IMDB Score'].min()


# In[61]:


lan = movie_info.groupby(['Language'])
lan.size()


# In[63]:


movie_info.groupby('Genre')[['Runtime']].apply(max)


# In[64]:


movie_info.groupby('Genre')[['Runtime', 'IMDB Score']].apply(max)


# In[69]:


#Aggregation
a = movie_info.groupby('Genre')['IMDB Score'].agg(['count','min','max','mean'])


# In[70]:


a


# In[73]:


movie_info.head(5)


# In[80]:


movie_info.groupby('Genre')['Runtime','IMDB Score'].agg(['min', 'max'])


# In[85]:


#COMBINE
movie_combine= movie_info.groupby('Genre').agg({'Runtime':['max','min'],'IMDB Score':['mean','max']})


# In[87]:


print(movie_combine)


# In[89]:


pd.DataFrame(movie_info.groupby('Genre').size(), columns=['Number'])


# In[ ]:




