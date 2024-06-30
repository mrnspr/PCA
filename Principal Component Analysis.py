#!/usr/bin/env python
# coding: utf-8

# # Principal Component Analysis
# 
# Let's discuss PCA! Notice PCA isn't exactly a full machine learning algorithm, but instead an unsupervised learning algorithm, 
# 
# Remember that PCA is just a transformation of your data and attempts to find out what features explain the most variance in your data. For example:

# In[16]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data
# 
# Let's work with the cancer data set again since it had so many features.

# In[17]:


from sklearn.datasets import load_breast_cancer


# In[18]:


cancer = load_breast_cancer()


# In[19]:


cancer.keys()


# In[20]:


print(cancer['DESCR'])


# In[21]:


df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
#(['DESCR', 'data', 'feature_names', 'target_names', 'target'])


# In[22]:


df.head()


# In[23]:


df.info()


# ## PCA Visualization
# 
# As we've noticed before it is difficult to visualize high dimensional data, we can use PCA to find the first two principal components, and visualize the data in this new, two-dimensional space, with a single scatter-plot. Before we do this though, we'll need to scale our data so that each feature has a single unit variance.

# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


scaler = StandardScaler()
scaler.fit(df)


# In[26]:


scaled_data=scaler.transform(df)


# In[27]:


scaled_data


# In[28]:


type(scaled_data)


# In[29]:


type(scaler)


# In[30]:


scaled_data.shape


# PCA with Scikit Learn uses a very similar process to other preprocessing functions that come with SciKit Learn. We instantiate a PCA object, find the principal components using the fit method, then apply the rotation and dimensionality reduction by calling transform().
# 
# We can also specify how many components we want to keep when creating the PCA object.

# In[31]:


from sklearn.decomposition import PCA


# In[32]:


pca = PCA(n_components=2)


# In[33]:


pca=pca.fit(scaled_data)


# In[34]:


x_pca=pca.transform(scaled_data)


# In[35]:


scaled_data.shape


# In[36]:


x_pca.shape


# Great! We've reduced 30 dimensions to just 2! Let's plot these two dimensions out!

# In[37]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[38]:


sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],hue=cancer['target'])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# Clearly by using these two components we can easily separate these two classes.
# 
# ## Interpreting the components 
# 
# Unfortunately, with this great power of dimensionality reduction, comes the cost of being able to easily understand what these components represent.
# 
# The components correspond to combinations of the original features, the components themselves are stored as an attribute of the fitted PCA object:

# In[40]:


pca.components_


# In[41]:


pca.components_.shape


# In this numpy matrix array, each row represents a principal component, and each column relates back to the original features. we can visualize this relationship with a heatmap:

# In[42]:


df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])


# In[43]:


df_comp


# In[44]:


plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)


# This heatmap and the color bar basically represent the correlation between the various feature and the principal component itself.
