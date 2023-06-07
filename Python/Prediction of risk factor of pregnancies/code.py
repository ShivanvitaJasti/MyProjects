#!/usr/bin/env python
# coding: utf-8

# In[283]:


# Loading the required libraries

import os
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

#reading CSV file into DataFrame 

dataset1 = pd.read_csv('C:/Users/Public/Dataset 1.csv') 

pd.set_option('display.max_columns', None)

dataset1.head(5)


# In[284]:


# splitting the column 'pastPregnancy' by > and storing it in a new dataframe

temp = dataset1['pastPregnancy'].str.split('>', expand=True)

# renaming required columns from index values

temp = temp.rename(columns={temp.columns[1]: 'A'})
temp = temp.rename(columns={temp.columns[2]: 'B'})
temp = temp.rename(columns={temp.columns[3]: 'C'})

# separating first character in the columns

temp['para'] = temp['A'].str[:1]
temp['living'] = temp['B'].str[:1]
temp['gravida'] = temp['C'].str[:1]

# cleaning the data

temp['para'] = temp['para'].str.replace('l','0')
temp['living'] = temp['living'].str.replace('g','0')
temp['gravida'] = temp['gravida'].str.replace('e','0')

# converting object type columns to integer type

temp['para'] = temp['para'].astype(str).astype(int)
temp['gravida'] = temp['gravida'].astype(str).astype(int)
temp['living'] = temp['living'].astype(str).astype(int)

# calculating gravida-para

dataset1['previous_unsuccessful_pregnancies'] = temp['gravida'] - temp['para']


# In[286]:


df2 = pd.read_csv('C:/Users/Public/dataset 2.csv')

df2.head()


# In[287]:


df3 = pd.read_csv('C:/Users/Public/dataset 3.csv',header=None)
col_names =["patient_id","appointment_type","appointment_status","clinical_observations_1","diagnosis_title","appointment_reason_1","start_time","end_time","info"]
df3.columns = col_names
df3.head()


# In[288]:


df2= df2.drop(["appointment_type","appointment_status","diagnosis_title","start_time","end_time"],axis=1)

df2.head()


# In[289]:


df3= df3.drop(["appointment_type","appointment_status","diagnosis_title","start_time","end_time"],axis=1)
df3.head()


# In[290]:


def concat_values(series):
    return ','.join(str(x) for x in series)

df2 = df2.groupby('patient_id')[['doctor_advice_notes', 'clinical_observations','appointment_reason']].agg(concat_values).reset_index()


# In[291]:


df3 = df3.groupby('patient_id')[[ 'clinical_observations_1','appointment_reason_1','info']].agg(concat_values).reset_index()


# In[292]:


df_merged = pd.merge(df2, df3, on='patient_id')


# In[293]:


df_merged.head(3)


# In[294]:


df_merged["full_clinical_obs"] =  df_merged['clinical_observations']+df_merged['clinical_observations_1']
df_merged["full_Appointment_reasons"] =  df_merged['appointment_reason']+df_merged['appointment_reason_1']
df_merged = df_merged.drop(["clinical_observations","clinical_observations_1","appointment_reason","appointment_reason_1"],axis=1)


# In[295]:


df =  pd.merge(dataset1,df_merged, on="patient_id",how ='outer')


# In[296]:


df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], format='%d/%m/%Y')
df['lmp'] = pd.to_datetime(df['lmp'], format='%d/%m/%Y')


# In[297]:


years, months = divmod((df['lmp'] - df['date_of_birth']).dt.days, 365)
df['Age_at_pregnancy'] = years.round()


# In[298]:


df['past_medical_history'] = df['past_medical_history'].apply(str)


# In[299]:


all_medical =''
for index, row in df.iterrows():
    if row['past_medical_history']!= "nan":
        all_medical += row['past_medical_history']
    all_medical += ','
  


# In[300]:


list1 = all_medical.split(",")
while '' in list1:
    list1.remove('')


# In[301]:


list1 = list(set(list1))


# In[302]:


list1.insert(0,'patient_id')
past_medical_df = pd.DataFrame(columns = (['patient_id']+list1))


# In[303]:


past_medical_df


# In[304]:


for index, row in df.iterrows():
    i =  row['past_medical_history']
    flag_list = []
    check_list = i.split(',')
    flag_list.append(row["patient_id"])
    for j in list1:
        if i =='nan':
            flag_list.append(0)
        elif j in check_list:
            flag_list.append(1)
        else:
            flag_list.append(0)
    past_medical_df = past_medical_df.append(pd.Series(flag_list, index = (['patient_id']+list1)), 
           ignore_index=True)        


# In[305]:


past_medical_df


# In[306]:


df = pd.merge(df, past_medical_df, on="patient_id", how= "left")


# In[310]:


df.isnull().sum()


# In[311]:


df.head(5)

df_SDG = df
df_SDG.head(5)


# In[312]:


categorical_features = ['medical_condition','co_morbidity','ethnicity','risk_factor','other checks','infection_history','past_medical_history','pastPregnancy','medications','doctor_advice_notes','info','full_clinical_obs','full_Appointment_reasons']

df_SDG = df_SDG.drop(['date_of_birth', 'lmp','edd'], axis=1)


# In[313]:


# ### CTGAN ###

# from ctgan import CTGAN


# In[314]:


# ctgan = CTGAN(verbose=True)
# ctgan.fit(df_SDG,categorical_features, epochs=200)


# In[315]:


# samples = ctgan.sample(1000)


# In[316]:


# samples.head(5)


# In[317]:


df_ML = df.drop(columns=['pastPregnancy','patient_id','date_of_birth','medical_condition','co_morbidity','ethnicity','lmp','edd','other checks','infection_history','past_medical_history','medications','doctor_advice_notes','info','full_clinical_obs','full_Appointment_reasons'])


# In[318]:


# samples = samples.drop(columns=['pastPregnancy','patient_id','medical_condition','co_morbidity','ethnicity','other checks','infection_history','past_medical_history','medications','doctor_advice_notes','info','full_clinical_obs','full_Appointment_reasons'])


# In[319]:


# samples.describe()


# In[320]:


# df_CTGANML = df_ML.append(samples, ignore_index=True)


# In[321]:


# df_CTGANML.describe()


# In[322]:


### SDV ###

from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df_SDG)

synthesizer = SingleTablePreset(
    metadata,
    name='FAST_ML'
)


synthesizer.fit(
    data=df_SDG
)
synthetic_data = synthesizer.sample(
    num_rows=5000
)

synthetic_data

encoding_synthetic_data = synthetic_data


# In[323]:


synthetic_data = synthetic_data.drop(columns=['pastPregnancy','patient_id','medical_condition','co_morbidity','ethnicity','other checks','infection_history','past_medical_history','medications','doctor_advice_notes','info','full_clinical_obs','full_Appointment_reasons'])


# In[324]:


# synthetic_data.to_csv(r'C:\Users\Public\synthetic_data.csv')


# In[325]:


df_encoding = df

df_encoding = df_encoding.append(encoding_synthetic_data, ignore_index=True)

# encoding ethnicity and medical condition columns to include in machine learning models

ethnicity = pd.get_dummies(df_encoding['ethnicity'],prefix_sep='_',prefix='eth')
medical_condition = pd.get_dummies(df_encoding['medical_condition'],prefix_sep='_',prefix='mc')

final_df = pd.concat([df_encoding, ethnicity],axis=1)
final_df = pd.concat([final_df, medical_condition],axis=1)

final_df = final_df.drop(columns=['lmp','edd','date_of_birth','pastPregnancy','patient_id','medical_condition','co_morbidity','ethnicity','other checks','infection_history','past_medical_history','medications','doctor_advice_notes','info','full_clinical_obs','full_Appointment_reasons'])


# In[326]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[327]:


predictor = final_df.drop('risk_factor', axis=1)
target = final_df[['risk_factor']]


# In[328]:


# splitting the data into test and train

predictor_train, predictor_test, target_train, target_test = train_test_split(predictor, target, test_size = 0.25, random_state=1)


# In[329]:


### DECISION TREE ###

decision_tree = DecisionTreeClassifier(criterion = "gini", random_state = 42,max_depth = 3, min_samples_leaf = 5)   
decision_tree = decision_tree.fit(predictor_train,target_train)

# criterion = "gini", random_state = 42,max_depth = 3, min_samples_leaf = 5


# In[330]:


target_predict_DT = decision_tree.predict(predictor_test)


# In[331]:


accuracy_score(target_test,target_predict_DT)


# In[334]:


# Visualizing the decision tree in the form of text

text_representation = tree.export_text(decision_tree)
print(text_representation)


# In[335]:


# Visualizing the decision tree

decision_tree_plot = plt.figure(figsize=(15,10))
tree.plot_tree(decision_tree)


# In[336]:


# Calculating the accuracy of the decision tree using 10-fold cross validation method:

from numpy import mean
from numpy import std

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv_dt = KFold(n_splits = 10, random_state = 1, shuffle = True)

scores_dt = cross_val_score(decision_tree, predictor, target, scoring = 'accuracy', cv = cv_dt, n_jobs = -1)

print('Accuracy: %.3f (%.3f)' % (mean(scores_dt), std(scores_dt)))


# In[337]:


### RANDOM FORESTS ###

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier 


# In[338]:


# Calculating the accuracy measure of the dataset using Random Forests with n_estimators = 10:

random_forest = RandomForestClassifier(n_estimators = 1000, random_state = 42)

random_forest.fit(predictor_train, target_train)

cv_rf = KFold(n_splits = 10, random_state = 1, shuffle = True)

scores_rf = cross_val_score(random_forest, predictor, target, scoring = 'accuracy', cv = cv_rf, n_jobs = -1)

print('Accuracy: %.3f (%.3f)' % (mean(scores_rf), std(scores_rf)))


# In[341]:


# ### GAUSSIAN NAIVE BAYES ###

# # importing the required libraries and fitting Gaussian NB to the dataset:

from sklearn.naive_bayes import GaussianNB

NB_classify = GaussianNB()
Gauss_NB = NB_classify.fit(predictor_train,target_train.values.ravel())
# NB_accuracy = Gauss_NB.score(predictor_test,target_predict)
# NB_accuracy


# In[342]:


# Using 10-fold cross validation method to calculate the accuracy:

GausNB_cv_score = cross_val_score(estimator = NB_classify, X = predictor_train,y = target_train.values.ravel(),cv = 10)
# print(GausNB_cv_score)

print("GaussianNB Cross Validation Score:", GausNB_cv_score.mean())


# In[343]:


### MULTINOMIAL NAIVE BAYES ###

# Using MinMaxScaler, as MultinomialNB does not accept negative values:

from sklearn.preprocessing import MinMaxScaler 

scaler1 = MinMaxScaler()
predictor_train = scaler1.fit_transform(predictor_train)
predictor_test = scaler1.transform(predictor_test)


# In[344]:


from sklearn.naive_bayes import MultinomialNB
MB_classify = MultinomialNB()
Multi_NB = MB_classify.fit(predictor_train,target_train.values.ravel())
MB_accuracy = Multi_NB.score(predictor_test,target_test)
MB_accuracy


# In[345]:


# Using 10-fold cross validation method to calculate the accuracy:

multiNB_cv_score = cross_val_score(estimator = MB_classify, X=predictor_train, y= target_train.values.ravel(),cv=10)
print("MultinomialNB Cross Validation Score:", multiNB_cv_score.mean())


# In[346]:


target_predict_mnb = Multi_NB.predict(predictor_test)
conf_mat_mnb = confusion_matrix(target_test, target_predict_mnb)
print(conf_mat_mnb)


# In[348]:


#### SVM ####

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

kernel = ['linear','sigmoid','poly','rbf']
for i in kernel:
    svc_model = SVC(kernel = i, C = 1)
    svc_model.fit(predictor_train,target_train)
    print('kernel:', i)
    print(svc_model.score(predictor_test,target_test))


# In[351]:


### VISUALIZATIONS ###

# ethnicity vs risk factor

import seaborn as sns

first_dimension = "risk_factor"
horizontal_label = "x label"
second_dimension = "ethnicity"

sns.histplot(binwidth=1,
            x=first_dimension,
            hue=second_dimension,
            data=df,
            stat="count",
            multiple="dodge")


# In[352]:


# Age at pregnancy vs risk factor

sns.boxplot(x='risk_factor', y='Age_at_pregnancy', data=df)


# In[353]:


complications = pd.read_csv('C:/Users/Public/complication.csv') 

df_viz = df.drop(columns=['lmp','edd','date_of_birth','pastPregnancy','patient_id','medical_condition','co_morbidity','other checks','infection_history','past_medical_history','medications'])

df_viz['gestational_diabetes'] = complications['gestational diabetes']
df_viz['preeclampsia'] = complications['pre-eclampsia']
df_viz['preterm'] = complications['pre-term labor']
df_viz['C-section'] = complications['unnecessary c-section']
df_viz['abnormal_weight'] = complications['abnormal weight']

df_viz.dtypes


# In[354]:


df_viz.head(10)


# In[355]:


import matplotlib.pyplot as plt


# In[356]:


sns.countplot(x ='gestational_diabetes', data = df_viz)

# Show the plot
plt.show()


# In[357]:


