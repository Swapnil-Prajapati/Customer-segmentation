#!/usr/bin/env python
# coding: utf-8

# # Importing neccessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# # Loading the data

# In[7]:


raw_data = pd.read_csv('Mall_Customers.csv')
raw_data.head()


# # Making copy of original data

# In[3]:


df = raw_data.copy()
df


# In[4]:


df.describe()


# In[11]:


plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'Gender' , data = df)
plt.show()


# In[18]:


labels = ['Female', 'Male']
size = df['Gender'].value_counts()
colors = ['orange', 'lightgreen']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()


# In[21]:


plt.figure(1 , figsize = (15,5))
sns.countplot(x = 'Age' , data = df)
plt.show()


# # clustering Results

# In[15]:


plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = df[df['Gender'] == gender] ,
                s = 200 , alpha = 0.75 , label = gender)
plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 
plt.title('Age vs Annual Income w.r.t Gender')
plt.legend()
plt.show()


# In[30]:


plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,
                data = df[df['Gender'] == gender] ,s = 200 , alpha = 0.75 , label = gender)
plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 
plt.title('Annual Income vs Spending Score w.r.t Gender')
plt.legend()
plt.show()


# # segmentation using Income and spending score

# In[31]:


x = raw_data[['Annual Income (k$)','Spending Score (1-100)']].copy()
x


# In[32]:


# standardizing 
from sklearn import preprocessing
x_scaled = preprocessing.scale(x)
x_scaled


# Now using elbow method to decise the no of clusters

# In[33]:


wcss =[]

for i in range(1,10):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
    
wcss


# In[35]:


plt.plot(range(1,10),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# The elbow shown here is at 3 and 5.
# let's try them both and see which ones's best suited

# # exploring the clusters

# In[41]:


#first we are going with 3

kmeans_new = KMeans(3)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)


# In[22]:


clusters_new.head


# In[44]:


plt.scatter(clusters_new['Annual Income (k$)'],clusters_new['Spending Score (1-100)'],
            c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')


# In[47]:


# we are going with 5

kmeans_new = KMeans(5)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)


# In[48]:


clusters_new


# In[49]:


plt.scatter(clusters_new['Annual Income (k$)'],clusters_new['Spending Score (1-100)'],
            c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')


# Rather than taking 3 centroids, taking 5 centroids looks more insightful and intuitive.
# We can consider other feature combinations as well (like gender and age) But annual income and spending score are perhaps having more variance in different clusters, so we will use this.

# In[ ]:





# In[ ]:




