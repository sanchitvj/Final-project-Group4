#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Loading Data
train_features = pd.read_csv('C:/Users/purvi/Downloads/MachineLearningMoAProject/train_features.csv')
train_targets = pd.read_csv('C:/Users/purvi/Downloads/MachineLearningMoAProject/train_targets_scored.csv')
test_features = pd.read_csv('C:/Users/purvi/Downloads/MachineLearningMoAProject/test_features.csv')


# Getting general understanding of the data 

# In[3]:


train_features.head()


# In[4]:


train_features.info()
train_features.describe()


# In[5]:


train_targets.head()


# In[6]:


train_targets.info()
train_targets.describe()


# In[7]:


test_features.head()


# In[8]:


test_features.info()
test_features.describe()


# # Checking for missing values in the datasets

# In[9]:


train_features.isnull().sum()


# In[10]:


train_targets.isnull().sum()


# In[11]:


test_features.isnull().sum()


# # Analyzing the distributions of numeric features using histograms and box plots
# 
# 

# In[12]:


# Calculate mean, median, and standard deviation for all numeric features
feature_mean = train_features.iloc[:, 4:].mean()
feature_median = train_features.iloc[:, 4:].median()
feature_std = train_features.iloc[:, 4:].std()

# Plot the distributions of summary statistics
plt.figure(figsize=(10, 6))
sns.histplot(feature_mean, bins=50, color='blue', kde=True, label='Mean')
sns.histplot(feature_median, bins=50, color='green', kde=True, label='Median')
sns.histplot(feature_std, bins=50, color='red', kde=True, label='Standard Deviation')
plt.xlabel('Summary Statistics')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# # Checking the correlation between the numeric features:
# 

# In[13]:


# Find the highest/lowest correlated feature pairs
corr_matrix = train_features.iloc[:, 4:].corr()
corr_matrix_flat = corr_matrix.unstack()
sorted_corr_matrix = corr_matrix_flat.sort_values(ascending=False, key=lambda x: abs(x))
highest_corr_pairs = sorted_corr_matrix[((sorted_corr_matrix != 1.0) & (sorted_corr_matrix != -1.0))].head(30).index.tolist()

# Create a new correlation matrix with only the highest/lowest correlated pairs
highest_corr_matrix = train_features[list(set([pair[0] for pair in highest_corr_pairs] + [pair[1] for pair in highest_corr_pairs]))].corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(highest_corr_matrix, cmap='crest', center=0, annot=True, fmt='.2f')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


# In[14]:



print(highest_corr_pairs)


# # Analyze the categorical features such as 'cp_type', 'cp_time', and 'cp_dose'

# In[15]:


# Count plots
plt.figure(figsize=(10, 6))
sns.countplot(x='cp_type', data=train_features)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='cp_time', data=train_features)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='cp_dose', data=train_features)
plt.show()


# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# 
# # PCA
# pca = PCA(n_components=2)
# train_features_pca = pca.fit_transform(train_features.iloc[:, 4:])
# 
# # t-SNE
# tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
# train_features_tsne = tsne.fit_transform(train_features.iloc[:, 4:])
# 
# # Visualization
# plt.figure(figsize=(10, 6))
# plt.scatter(train_features_pca[:, 0], train_features_pca[:, 1], s=5, alpha=0.5)
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.show()
# 
# plt.figure(figsize=(10, 6))
# plt.scatter(train_features_tsne[:, 0], train_features_tsne[:, 1], s=5, alpha=0.5)
# plt.xlabel('t-SNE1')
# plt.ylabel('t-SNE2')
# plt.show()
# 
# 

# In[16]:


#Outliers (using IQR method):
Q1 = train_features.iloc[:, 4:].quantile(0.25)
Q3 = train_features.iloc[:, 4:].quantile(0.75)
IQR = Q3 - Q1

outliers = ((train_features.iloc[:, 4:] < (Q1 - 1.5 * IQR)) | (train_features.iloc[:, 4:] > (Q3 + 1.5 * IQR))).sum()
print(outliers[outliers > 0])


# In[17]:


#Feature engineering:
train_features['sum_g'] = train_features.iloc[:, 4:4+772].sum(axis=1)
print(train_features[['sig_id', 'sum_g']].head())


# In[18]:


#Feature scaling (using StandardScaler):
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(train_features.iloc[:, 4:])
scaled_features_df = pd.DataFrame(scaled_features, columns=train_features.columns[4:])


# In[19]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

# Train the OneVsRestClassifier with RandomForestClassifier
ovr_rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
ovr_rf.fit(train_features.iloc[:, 4:], train_targets.iloc[:, 1:])

# Calculate the average feature importances from all the RandomForestClassifiers
feature_importances = np.mean([estimator.feature_importances_ for estimator in ovr_rf.estimators_], axis=0)

# Sort the features by importance
importances = pd.DataFrame({'feature': train_features.columns[4:], 'importance': feature_importances})
importances = importances.sort_values('importance', ascending=False)

print(importances.head(10))



# In[ ]:





# In[20]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_features_scaled = train_features.copy()
train_features_scaled.iloc[:, 4:] = scaler.fit_transform(train_features.iloc[:, 4:])


# In[22]:


import matplotlib.pyplot as plt

# Plot the distribution of each target variable in train_targets_scored
fig, ax = plt.subplots(figsize=(15, 30))  # Increase the figure size
train_targets.sum(axis=0)[1:].sort_values().plot(kind='barh', ax=ax)  # Change the plot to horizontal
plt.ylabel('Targets')
plt.xlabel('Frequency')
plt.title('Distribution of Target Variables')
plt.tight_layout()
plt.show()


# In[ ]:




