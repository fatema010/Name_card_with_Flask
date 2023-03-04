#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


raw_csv_data = pd.read_csv('C:/Users/fatem/OneDrive/Desktop/PROJECT/Absenteeism_data.csv')
raw_csv_data


# In[3]:


df = raw_csv_data.copy()


# In[4]:


df


# In[5]:


display(df)


# In[6]:


type(df)


# In[7]:


df = df.drop(['ID'],axis = 1)


# In[8]:


df 


# In[9]:


pd.unique(df['Reason for Absence'])


# In[10]:


df.copy()


# In[11]:


reason_columns = pd.get_dummies(df['Reason for Absence'])


# In[12]:


reason_columns


# In[13]:


reason_columns['check'] = reason_columns.sum(axis=1)


# In[14]:


reason_columns


# In[15]:


reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)


# In[16]:


reason_columns


# In[17]:


df.columns.values


# In[18]:


reason_columns.columns.values


# In[19]:


df


# In[20]:


df = df.drop(['Reason for Absence'],axis = 1)


# In[21]:


df


# In[ ]:





# In[22]:


df.columns.values


# In[23]:


reason_columns.columns.values


# In[24]:


df


# In[25]:


reason_columns.loc[:, 1:14].max(axis=1)
reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)    


# In[26]:


reason_type_1


# In[27]:


reason_type_2


# In[28]:


reason_type_3


# In[29]:


reason_type_4


# In[30]:


df


# In[31]:


df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)


# In[32]:


df.columns.values


# In[33]:


df.columns.values


# In[34]:


column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age','Daily Work Load Average', 'Body Mass Index', 'Education',
'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']


# In[35]:


df.columns = column_names


# In[36]:


df.head()


# In[37]:


column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
    'Date', 'Transportation Expense', 'Distance to Work', 'Age',
    'Daily Work Load Average', 'Body Mass Index', 'Education','Children', 'Pets', 'Absenteeism Time in Hours']


# In[38]:


df = df[column_names_reordered]


# In[39]:


df.head()


# In[40]:


df_reason_mod = df.copy()


# In[41]:


df_reason_mod


# In[42]:


type(df_reason_mod['Date'][0])


# In[43]:


df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'],format = '%d/%m/%Y')


# In[44]:


df_reason_mod['Date']


# In[45]:


df_reason_mod['Date'][0].month


# In[46]:


list_months = []
list_months


# In[47]:


df_reason_mod.shape


# In[ ]:





# In[48]:


for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)


# In[49]:


list_months


# In[50]:


len(list_months)


# In[ ]:





# In[51]:


df_reason_mod['Month value'] = list_months


# In[52]:


df_reason_mod.head()


# In[53]:


def date_to_weekday(date_value):
    return date_value.weekday()


# In[54]:


df_reason_mod['Day of the week'] = df_reason_mod['Date'].apply(date_to_weekday)


# In[55]:


df_reason_mod.head()


# In[56]:


df_reason_mod = df_reason_mod.drop(['Date'],axis = 1)


# In[57]:


df_reason_mod


# In[58]:


df_reason_mod.columns.values


# In[59]:


df_reason_new = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Month value',
       'Day of the week',
       'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours'] 


# In[60]:


df_reason_mod = df_reason_mod[df_reason_new]


# In[61]:


df_reason_mod


# In[62]:


df_reason_mod_new = df_reason_mod.copy()


# In[63]:


df_reason_mod_new


# In[64]:


type(df_reason_mod_new['Transportation Expense'][0])


# In[65]:


type(df_reason_mod_new['Distance to Work'][0])


# In[66]:


type(df_reason_mod_new['Age'][0])


# In[67]:


df_reason_mod_new['Education'].unique()


# In[68]:


df_reason_mod_new['Education'].value_counts()


# In[69]:


df_reason_mod_new['Education'] = df_reason_mod_new['Education'].map({1:0,2:1,3:1,4:1})


# In[70]:


df_reason_mod_new['Education'].unique()


# In[71]:


df_reason_mod_new['Education'].value_counts()


# In[72]:


df_preprocessed = df_reason_mod_new.copy()


# In[73]:


df_preprocessed


# In[74]:


df_preprocessed.to_csv('C:/Users/fatem/OneDrive/Desktop/PROJECT/Absenteeism_preprocessed_data.csv', index=False)


# In[75]:


df_preprocessed


# In[ ]:




