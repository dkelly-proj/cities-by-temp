#!/usr/bin/env python
# coding: utf-8

# ### Import modules

# In[1]:


import pandas as pd
import requests


# ### Get data from wikipedia

# In[13]:


url = 'https://en.wikipedia.org/wiki/List_of_cities_by_average_temperature'

data = pd.concat(pd.read_html(requests.get(url).content)).reset_index(drop = True)


# ### Clean and organize data

# In[50]:


df = (data
          .melt(id_vars = ['Country', 'City'], value_vars = ['Jan', 'Feb', 'Mar', 'Apr',
                                                             'May', 'Jun', 'Jul', 'Aug',
                                                             'Sep', 'Oct', 'Nov', 'Dec'],
                var_name = 'Month', value_name = 'Temp_String'))

df['Temp_F'] = df.Temp_String.str.split(pat = '(', expand = True)[1].str.split(pat = ')', expand = True)[0]
df['City'] = df['City'] + ', ' + df['Country']

df = (df
          .filter(items = ['City', 'Temp_F'])
          .groupby(['City'])
          .agg(['min', 'max'])
          .reset_index()
          .droplevel(0, axis = 1))

df.columns = ['City', 'Min', 'Max']


# In[ ]:


print(df.sort_values('Min'))

