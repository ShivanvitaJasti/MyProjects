#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading the required libraries

import os
import pandas as pd 
import matplotlib.pyplot as plot 
import numpy as np

#reading CSV file into DataFrame 

Development_dataset = pd.read_csv('C:/Users/Public/Development.csv') 

pd.set_option('display.max_columns', None)

Development_dataset.head(5)


# In[2]:


# Setting the working directory 

os.chdir('/Users/Public')
os.getcwd()


# In[3]:


# Displaying the column names in the dataset

Development_dataset.columns


# In[4]:


# Displaying the data types of the variables in the dataset

Development_dataset.dtypes


# In[5]:


## Cleaning the data:
### Creating a subset of the dataset by omitting the variables that are not significant for the analysis:

development_subset = Development_dataset[['SqFootage', 'ProjType', 'Rooms',  'AppType',
  'ProjName', 'Category', 'CreYear', 'LastYear', 'BOZA' , 'PC', 'LDT', 'DRC', 'Units']]

development_subset.head(5)


# In[6]:


# Checking for null values in the dataset:

development_subset.isna().sum()


# In[7]:


# Filling the null values in the BOZA, PC, LDT and DRC columns with ‘No Decision’, 
# as a null value in these columns suggests that a decision has not yet been made by the committees on the project:

values = {"BOZA": "No decision","PC": "No decision","LDT": "No decision","DRC": "No decision"}
development_subset = development_subset.fillna(value = values)


# In[8]:


# Displaying the first few records in the dataset after filling in the missing values:

development_subset.head(5)


# In[9]:


# Checking for null values in the dataset:

development_subset.isna().sum()


# In[10]:


# Viewing the row where ‘ProjName’ variable has a null value

development_subset[development_subset['ProjName'].isna()]


# In[11]:


# Filling the null value with 'Uncertain'

values1 = {"ProjName": "Uncertain"}
development_subset = development_subset.fillna(value = values1)


# In[12]:


# viewing the row that was filled in the previous step:

development_subset.loc[development_subset['SqFootage'] == 19530]


# In[13]:


# Viewing the instances in the Project_Name column:

development_subset.ProjType.value_counts()


# In[14]:


# Viewing the counts of Zoning Projects:

development_subset['ProjType'].str.contains('zoned', case = False).value_counts()


# In[15]:


#combining all the zoning projects into one Project Type:

development_subset.loc[development_subset['ProjType'].str.contains('zoned', case=False), 'ProjType'] = 'Zoned'


# In[16]:


# viewing the size of the cleaned dataset:

development_subset.shape


# In[17]:


# Viewing the datatypes and the non-null counts of the cleaned dataset:

print(development_subset.info())


# In[18]:


# Describing the summary statistics for the numerical variables in the dataset:

development_subset.describe()


# In[27]:


# Exporing the cleaned dataset to a csv file, for the purpose of analysis in R and SQL:

development_subset.to_csv('Cleaned_Development_data1.csv')


# In[180]:


# 1. How many number of projects are there per Project type?

pd.set_option("display.max_rows", None)
development_subset.ProjType.value_counts()


# In[181]:


plot.figure(figsize=(20,20))
ax = sns.countplot(data = development_subset,y = 'ProjType')
for i in ax.containers:
    ax.bar_label(i,)


# In[182]:


# 2. Identify the number of projects per Application Type and Category.

pd.set_option("display.max_rows", None)
development_subset.AppType.value_counts()


# In[186]:


AppType = development_subset['AppType'].value_counts()
sns.set(style="darkgrid")
ax = sns.barplot(AppType.index, AppType.values, alpha=0.9)
for i in ax.containers:
    ax.bar_label(i,)
plot.title('Frequency Distribution of Application Type')
plot.ylabel('Number of Instances', fontsize=12)
plot.xlabel('Application Type', fontsize=12)
plot.xticks(rotation = 45)
plot.show()


# In[183]:


pd.set_option("display.max_rows", None)
development_subset.Category.value_counts()


# In[187]:


Category = development_subset['Category'].value_counts()
sns.set(style="darkgrid")
ax = sns.barplot(Category.index, Category.values, alpha=0.9)
for i in ax.containers:
    ax.bar_label(i,)
plot.title('Frequency Distribution of Category')
plot.ylabel('Number of Instances', fontsize=12)
plot.xlabel('Category', fontsize=12)
plot.xticks(rotation = 45)
plot.show()


# In[193]:


# 3. Which category and Application Type have the highest number of projects?

pd.set_option("display.max_rows", None)
development_subset.AppType.value_counts().head(1)


# In[194]:


pd.set_option("display.max_rows", None)
development_subset.Category.value_counts().head(1)


# In[23]:


# 4. List and count of projects that are hospitals.

development_subset[development_subset['ProjName'].str.contains("hospital | hospital| hospital ",case=False)]


# In[196]:


# 6. Which project has highest square footage and highest number of rooms?

Area_max = development_subset['SqFootage'].max()
development_subset.loc[development_subset['SqFootage'] == Area_max]


# In[197]:


Rooms_max = development_subset['Rooms'].max()
development_subset.loc[development_subset['Rooms'] == Rooms_max]


# In[26]:


# 7. How many projects are proposed to be built on a highway?

development_subset[development_subset['ProjName'].str.contains("highway | highway| highway ",case=False)]


# In[17]:


import seaborn as sns


# In[24]:


# plot.figure(figsize=(40,20))
# development_subset.plot.scatter(x='Units',y='ProjType',c='DarkBlue')

plot.figure(figsize=(20,15))
plot.scatter(development_subset.Units, development_subset.ProjType, marker = "*", color = 'brown')
plot.xlabel('Number of Multi-Family dwelling units' , fontsize = 15)
plot.ylabel('Project Type', fontsize = 15)
plot.title('Relation between the type of Project and the number of multi-family dwelling units in the Project',fontsize = 20)
plot.grid(b = True , color = 'grey')
plot.show()

