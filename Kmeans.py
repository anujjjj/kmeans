
# coding: utf-8

# In[1]:


import sys
print(sys.version)


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[32]:


#Generate Data -two means
mean1 = [np.random.randint(50),np.random.randint(50)]
mean2 = [np.random.randint(50),np.random.randint(50)]

#Make diagonal covariance
cov=[[100,0],[0,100]]

x1,y1 = np.random.multivariate_normal(mean1,cov,100).T
x2,y2 = np.random.multivariate_normal(mean2,cov,100).T

x= np.append(x1,x2)
y=np.append(y1,y2)

plt.plot(x,y,'x')
plt.axis('equal')
plt.show()


# In[33]:


X= (zip(x,y))
X=list(X)


# In[34]:


#Make K means model
kmeans = KMeans(n_clusters=2)


# In[35]:


#Fit the model to data
kmeans.fit(X)


# In[36]:


centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(labels)


# In[37]:


print(centroids)


# In[39]:


colors = ["g.","r."]
for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]], markersize=10)

plt.scatter(centroids[:,0],centroids[:,1],marker = "X" , s=250 , zorder=10)

plt.show()


# In[40]:


print(centroids)
print(mean1,mean2)

