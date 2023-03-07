#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[3]:


data = pd.read_csv("C:/Users/fatem/OneDrive/Desktop/PROJECT/heart_failure.csv")


# In[4]:


type(data)


# In[5]:


data.shape


# In[6]:


data


# In[7]:


data.columns.values


# In[8]:


pd.set_option('display.max_rows',300)
data


# In[9]:


categorical_variables = data[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']]
continuous_variables = data[['age', 'creatinine_phosphokinase',
       'ejection_fraction', 'platelets',
       'serum_creatinine', 'serum_sodium', 'time',
       'DEATH_EVENT']]


# In[10]:


data


# Note:
# anaemia: 0 means that the person does not have anaemia, if 1 it does.
# diabetes: 0 means that the person does not have diabetes, if 1 it does.
# high_blood_pressure:0 means that the person does not have, if 1 it has.
# smoking: 0 means that the person does not smoke, if 1 it smokes.

# In[11]:


type(categorical_variables)


# In[12]:


pd.set_option('display.max_rows',300)
data.isna()


# As we can see there is no missing data in our dataset.

# In[13]:


data.isna().sum()


# In[14]:


data.isnull().sum()
data.isnull()


# There is no null values in this dataset

# In[15]:


continuous_variables.describe()


# describe() function helps with the descriptive statistics.describe function works only for continuous_variables.

# In[16]:


data.groupby('DEATH_EVENT').count()


# The data shows the dataset is unbalanced,because the total death(96) is less than half of the total persons who did not die because of heart failure.

# In[17]:


age = data[['age']]
platelets = data[['platelets']]


# In[18]:


type(data[['platelets']])


# In[19]:


type(data['age'])


# In[20]:


plt.figure(figsize=(15,7))
plt.scatter(platelets, age, c = data['DEATH_EVENT'],s= 200,alpha=0.7)
plt.xlabel('platelets', fontsize=22)
plt.ylabel('Age',fontsize=22)
plt.title('Showing the unbalanced data',fontsize=25)
plt.show()


# showing the data with a scatter plot,the colors will depends on the DEATH_EVENT(0, 1). Also more circles in purple than in yellow.

# In[21]:


plt.figure(figsize=(14,6))
sns.heatmap(data.corr(),vmin = -1, vmax = 1, cmap = 'BrBG_r', annot = True)
plt.title ('Relationship between all the variables and DEATH_EVENT',fontsize = 20)
plt.show()


# There is a positive correlation between DEATH_EVENT and serum creatinine and age.Negetive correlation between DEATH_EVENT and time, ejection fraction and serum sodium.
# 

# In[22]:


plt.figure(figsize=(14,10))
for i,cat in enumerate(categorical_variables):
    plt.subplot(2,3,i+1)
    sns.countplot(data = data,x= cat,hue = "DEATH_EVENT")
plt.show()


# plotting the impact of categorical variables on DEATH_EVENT.
# 

# In[23]:


plt.figure(figsize=(14,10))
for q,con in enumerate(continuous_variables):
    plt.subplot(3,3,q+1)
    sns.histplot(data = data,x= con,hue = "DEATH_EVENT",multiple = 'stack')
plt.show()


# In[24]:


plt.figure(figsize=(10,10))
sns.boxplot(data = data,x= 'sex',y='age',hue = "DEATH_EVENT")
plt.title('The influence of sex and age on the death event')
plt.show()


# Survival status on smoking.

# In[25]:


smokers = data[data['smoking']==1]
non_smokers = data[data['smoking']==0]

non_survived_smokers= smokers[smokers['DEATH_EVENT']==1]
survived_non_smokers= non_smokers[non_smokers['DEATH_EVENT']==0]
non_survived_non_smokers= non_smokers[non_smokers['DEATH_EVENT']==1]
survived_smokers= smokers[smokers['DEATH_EVENT']==0]

smoking_data = [len(non_survived_smokers),len(survived_non_smokers),len(non_survived_non_smokers),len(survived_smokers)]
smoking_labels = ['non_survived_smokers','survived_non_smokers','non_survived_non_smokers','survived_smokers']
plt.figure(figsize=(10,10))
plt.pie(smoking_data,labels = smoking_labels,autopct='%.1f%%',startangle=90)
circle=plt.Circle((0,0),0.7,color='gray')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('survival status on smoking',fontsize=22)
plt.show()


# gcf= get current figure. gca= get current access

# In[26]:


plt.pie(smoking_data,labels = smoking_labels,autopct='%.1f%%',startangle=90)


# Analyzing the survival status on sex.

# In[27]:


male = data[data['sex']==1]
female = data[data['sex']==0]

non_survived_male= male[male['DEATH_EVENT']==1]
survived_male= male[male['DEATH_EVENT']==0]
non_survived_female= female[female['DEATH_EVENT']==1]
survived_female= female[female['DEATH_EVENT']==0]

sex_data = [len(non_survived_male),len(survived_male),len(non_survived_female),len(survived_female)]
sex_labels = ['non_survived_male','survived_male','non_survived_female','survived_female']
plt.figure(figsize=(10,10))
plt.pie(sex_data,labels = sex_labels,autopct='%.1f%%',startangle=90)
circle=plt.Circle((0,0),0.7,color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('survival status on sex',fontsize=22)
plt.show()


# In[28]:


with_diabetes = data[data['diabetes']==1]
without_diabetes = data[data['diabetes']==0]

non_survived_with_diabetes= with_diabetes[with_diabetes['DEATH_EVENT']==1]
survived_with_diabetes= with_diabetes[with_diabetes['DEATH_EVENT']==0]
non_survived_without_diabetes= without_diabetes[without_diabetes['DEATH_EVENT']==1]
survived_without_diabetes= without_diabetes[without_diabetes['DEATH_EVENT']==0]

diabetes_data = [len(non_survived_with_diabetes),len(survived_with_diabetes),len(non_survived_without_diabetes),len(survived_without_diabetes)]
diabetes_labels = ['non_survived_with_diabetes','survived_with_diabetes','non_survived_without_diabetes','survived_without_diabetes']
plt.figure(figsize=(10,10))
plt.pie(diabetes_data,labels = diabetes_labels,autopct='%.1f%%',startangle=90)
circle=plt.Circle((0,0),0.7,color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('survival status on sdiabetes',fontsize=22)
plt.show()


# In[29]:


with_anaemia = data[data['anaemia']==1]
without_anaemia = data[data['anaemia']==0]

non_survived_with_anaemia= with_anaemia[with_anaemia['DEATH_EVENT']==1]
survived_with_anaemia= with_anaemia[with_anaemia['DEATH_EVENT']==0]
non_survived_without_anaemia= without_anaemia[without_anaemia['DEATH_EVENT']==1]
survived_without_anaemia= without_anaemia[without_anaemia['DEATH_EVENT']==0]

anaemia_data = [len(non_survived_with_anaemia),len(survived_with_anaemia),len(non_survived_without_anaemia),len(survived_without_anaemia)]
anaemia_labels = ['non_survived_with_anaemia','survived_with_anaemia','non_survived_without_anaemia','survived_without_anaemia']
plt.figure(figsize=(10,10))
plt.pie(anaemia_data,labels = anaemia_labels,autopct='%.1f%%',startangle=90)
circle=plt.Circle((0,0),0.7,color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('survival status on anaemia',fontsize=22)
plt.show()


# In[30]:


with_high_blood_pressure = data[data['high_blood_pressure']==1]
without_high_blood_pressure = data[data['high_blood_pressure']==0]

non_survived_with_high_blood_pressure= with_high_blood_pressure[with_high_blood_pressure['DEATH_EVENT']==1]
survived_with_high_blood_pressure= with_high_blood_pressure[with_high_blood_pressure['DEATH_EVENT']==0]
non_survived_without_high_blood_pressure= without_high_blood_pressure[without_high_blood_pressure['DEATH_EVENT']==1]
survived_without_high_blood_pressure= without_high_blood_pressure[without_high_blood_pressure['DEATH_EVENT']==0]

high_blood_pressure_data = [len(non_survived_with_high_blood_pressure),len(survived_with_high_blood_pressure),len(non_survived_without_high_blood_pressure),len(survived_without_high_blood_pressure)]
high_blood_pressure_labels = ['non_survived_with_high_blood_pressure','survived_with_high_blood_pressure','non_survived_without_high_blood_pressure','survived_without_high_blood_pressure']
plt.figure(figsize=(10,10))
plt.pie(high_blood_pressure_data,labels = high_blood_pressure_labels,autopct='%.1f%%',startangle=90)
circle=plt.Circle((0,0),0.7,color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('survival status on high_blood_pressure',fontsize=22)
plt.show()


# In[ ]:




