#!/usr/bin/env python
# coding: utf-8

# ### Import modules

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import requests
from sklearn.cluster import KMeans


# ### Get data from wikipedia

# In[ ]:


url = 'https://en.wikipedia.org/wiki/List_of_cities_by_average_temperature'

data = pd.concat(pd.read_html(requests.get(url).content)).reset_index(drop = True)


# ### Clean and organize data

# In[ ]:


df = (data
          .melt(id_vars = ['Country', 'City'], value_vars = ['Jan', 'Feb', 'Mar', 'Apr',
                                                             'May', 'Jun', 'Jul', 'Aug',
                                                             'Sep', 'Oct', 'Nov', 'Dec'],
                var_name = 'Month', value_name = 'Temp_String'))

df['Temp_F'] = df.Temp_String.str.split(pat = '(', expand = True)[1].str.split(pat = ')', expand = True)[0]
df['City'] = df['City'] + ', ' + df['Country']
df['Temp_F'] = pd.to_numeric(df['Temp_F'], errors = 'coerce')

df = (df
          .filter(items = ['City', 'Temp_F'])
          .groupby(['City'])
          .agg(['min', 'max'])
          .reset_index()
          .droplevel(0, axis = 1)
          .query('min == min and max == max'))

df.columns = ['City', 'Min', 'Max']


# In[ ]:


print(df)


# ### Identify optimal number of clusters for K Means

# In[ ]:


temp_pairs = np.array(df[['Min', 'Max']])

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(temp_pairs)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('Elbow WCSS')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


wcss_per = [(wcss[i] - wcss[i-1])/wcss[i-1] for i in range(1,10)]
print(pd.Series(wcss_per))

clusters = int(input('How many clusters should we use?: '))


# ### Assign clusters

# In[ ]:


kmeans = KMeans(n_clusters = clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
df['Cluster'] = kmeans.fit_predict(temp_pairs)


# ### Plot

# In[ ]:


palette = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'purple']

plt.figure(figsize = (20,10))

for i in range(0, max(df['Cluster']) + 1):
    plt.scatter(df[df['Cluster'] == i]['Min'], df[df['Cluster'] == i]['Max'], c = palette[i], s = 300)

plt.title('Cities Clustered by Highest and Lowest Monthly Average Temperatures Fahrenheit')
plt.xlabel('Lowest Monthly Average Temperature')
plt.ylabel('Highest Monthly Average Temperature')
plt.savefig('City_Temp_Clusters.png')


# ### Open and view plot

# In[ ]:


img = Image.open('City_Temp_Clusters.png')
img.show()

