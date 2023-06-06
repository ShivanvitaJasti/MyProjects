#!/usr/bin/env python
# coding: utf-8

# In[25]:


# Loading the required libraries

import os
import pandas as pd 
import matplotlib.pyplot as plot 
import numpy as np

#reading CSV file into DataFrame 

Student_data = pd.read_csv('C:/Users/Public/Student_Behaviour.csv') 

pd.set_option('display.max_columns', None)

Student_data.head(5)


# In[26]:


# Setting the working directory 

os.chdir('/Users/Public')
os.getcwd()


# In[28]:


# displaying column names to rename complex attribute names:

for column in Student_data.columns:
    print(column)


# In[27]:


# Renaming complex attribute names:

Student_data.rename(columns = {'Have you completed any certification courses, or are you currently enrolled in any?':'Certifications'}, inplace = True)
Student_data.rename(columns = {'daily studing time':'Daily_study_time'}, inplace = True)
Student_data.rename(columns = {'prefer to study in':'Prefer_study_time'}, inplace = True)
Student_data.rename(columns = {'salary expectation':'Exp_salary'}, inplace = True)
Student_data.rename(columns = {'Do you like your degree?':'Interest'}, inplace = True)
Student_data.rename(columns = {'possibility of choosing  their career based on their degree : ':'Career_basedon_degree'}, inplace = True)


# In[29]:


## Processing the data:
### Creating a subset of the dataset by omitting the variables that are not significant for the analysis:

Student_subset = Student_data[['Certifications', '10th Mark', '12th Mark',  'college mark',
  'hobbies', 'Daily_study_time', 'Prefer_study_time', 'Exp_salary', 'Interest' , 
  'Career_basedon_degree', 'Stress Level ', 'Financial Status']]

Student_subset.head(5)


# In[30]:


for column in Student_subset.columns:
    print(column)


# In[31]:


# Exporing the processed dataset to a csv file, for the purpose of analysis:

Student_subset.to_csv('Cleaned_Student_data.csv')

