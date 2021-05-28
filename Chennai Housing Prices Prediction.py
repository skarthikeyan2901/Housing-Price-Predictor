#!/usr/bin/env python
# coding: utf-8

# ### Chennai Housing Prices Prediction

# The dataset contains information related to all real estate transactions that have taken place under the ChennaiEstate. The following are the details relating to the columns in the dataset:
# 
# PRTID – The Property Transaction ID assigned by Chennai Estate AREA – The property in which the real estate is located INTSQFT – The interior Sq. Ft of the property
# DATESALE – The date the property was sold DISTMAINROAD – The distance of the property to the main road
# NBEDROOM – The number of Bedrooms NBATHROOM - The number of bathrooms
# NROOM – Total Number of Rooms SALECOND – The Sale Condition
# Normal: Normal Sale
# Abnormal: Abnormal Sale - trade, foreclosure, short sale
# AdjLand: Adjoining Land Purchase
# Family: Sale between family members
# Partial: Home was not completed when last assessed
# PARKFACIL – Whether parking facility is available DATEBUILD – The date in which the property was built
# BUILDTYPE – The type of building
# House
# Commercial
# Others
# UTILITYAVAIL AllPub: All public Utilities (E,G,W,& S) NoSewr: Electricity, Gas, and Water (Septic Tank) NoSeWa: Electricity and Gas Only ELO: Electricity only STREET Gravel Paved No Access MZZONE A: Agriculture C: Commercial I: Industrial RH: Residential High Density RL: Residential Low Density RM: Residential Medium Density QSROOMS – The quality score assigned for rooms based on buyer reviews
# QSBATHROOM – The quality score assigned for bathroom based on buyer reviews QSBEDROOM – The quality score assigned for bedroom based on buyer reviews
# QSOVERALL – The Overall quality score assigned for the property REGFEE – The registration fee for the property
# COMMIS – The Commission paid to the agent
# SALES_PRICE – The total sale price of the property

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("D:/Online Course/Projects/Datasets/Chennai Housing Prices/train.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


plt.figure(figsize = (12,6))
sns.heatmap(df.isnull(),cmap='coolwarm')


# In[8]:


df.corr()['SALES_PRICE'].sort_values(ascending=False)


# In[9]:


df.corr()


# In[10]:


df['SALE_COND'].value_counts()


# In[11]:


df.shape


# In[12]:


df.columns


# In[13]:


for i in range(len(df)):
    if (pd.isnull(df['QS_OVERALL'][i])==True):
        df['QS_OVERALL'][i] = (df['QS_BEDROOM'][i] + df['QS_BATHROOM'][i] + df['QS_ROOMS'][i]) / 3


# In[14]:


df.isnull().sum()


# In[15]:


df.dropna(inplace=True)


# In[16]:


#null_index = df[df['QS_OVERALL'].isna()].index.values


# In[17]:


#df['QS_OVERALL'] = (df['QS_BEDROOM'] + df['QS_BATHROOM'] + df['QS_ROOMS']) / 3  


# In[18]:


df.shape


# In[19]:


df.isnull().sum()


# In[20]:


park_facil={'Yes':1,'No':0}
df['PARK_FACIL'] = df['PARK_FACIL'].map(park_facil)


# In[21]:


df.head()


# In[22]:


plt.figure(figsize = (12,7))
sns.scatterplot(x='INT_SQFT',y='SALES_PRICE', data=df)


# In[23]:


sns.barplot(df['N_BEDROOM'],df['SALES_PRICE'],data=df,hue=df['PARK_FACIL'])


# In[24]:


plt.figure(figsize=(15,9))
sns.barplot(df['AREA'],df['SALES_PRICE'],data=df)
plt.tight_layout()


# In[25]:


df['AREA'].value_counts()


# In[26]:


df.loc[df['AREA'] == 'Chrompt','AREA'] = 'Chrompet'


# In[27]:


df.loc[df['AREA'] == 'Chrompt','AREA'] = 'Chrompet'
df.loc[df['AREA'] == 'Chrmpet','AREA'] = 'Chrompet'
df.loc[df['AREA'] == 'Chormpet','AREA'] = 'Chrompet'
df.loc[df['AREA'] == 'TNagar','AREA'] = 'T Nagar'
df.loc[df['AREA'] == 'Ana Nagar','AREA'] = 'Anna Nagar'
df.loc[df['AREA'] == 'Ann Nagar','AREA'] = 'Anna Nagar'
df.loc[df['AREA'] == 'Karapakam','AREA'] = 'Karapakkam'
df.loc[df['AREA'] == 'Velchery','AREA'] = 'Velachery'
df.loc[df['AREA'] == 'KKNagar','AREA'] = 'KK Nagar'
df.loc[df['AREA'] == 'Adyr','AREA'] = 'Adyar'


# In[28]:


df['AREA'].value_counts()


# In[29]:


plt.figure(figsize=(12,7))
sns.barplot(df['AREA'],df['SALES_PRICE'],data=df)
plt.tight_layout()


# In[30]:


for i in df.columns.values:
    print(df[i].value_counts())
    print("\n\n\n\n")


# In[31]:


df.loc[df['SALE_COND'] == 'Partiall','SALE_COND'] = 'Partial'
df.loc[df['SALE_COND'] == 'PartiaLl','SALE_COND'] = 'Partial'
df.loc[df['SALE_COND'] == 'Ab Normal','SALE_COND'] = 'AbNormal'
df.loc[df['SALE_COND'] == 'Adj Land','SALE_COND'] = 'AdjLand'


# In[32]:


df.loc[df['BUILDTYPE'] == 'Comercial','BUILDTYPE'] = 'Commercial'
df.loc[df['BUILDTYPE'] == 'Other','BUILDTYPE'] = 'Others'


# In[33]:


df.loc[df['UTILITY_AVAIL'] == 'All Pub','UTILITY_AVAIL'] = 'AllPub'


# In[34]:


df.loc[df['STREET'] == 'Pavd','STREET'] = 'Paved'
df.loc[df['STREET'] == 'NoAccess','STREET'] = 'No Access'


# In[35]:


for i in df.columns.values:
    print(df[i].value_counts())
    print("\n\n\n\n")


# In[36]:


sns.barplot(df['STREET'],df['SALES_PRICE'],data=df)


# In[37]:


sns.barplot(df['SALE_COND'],df['SALES_PRICE'],data=df)


# In[38]:


sns.barplot(df['BUILDTYPE'],df['SALES_PRICE'],data=df)


# In[39]:


sns.barplot(df['UTILITY_AVAIL'],df['SALES_PRICE'],data=df)


# In[40]:


sns.barplot(df['MZZONE'],df['SALES_PRICE'],data=df)


# In[41]:


df.corr()['SALES_PRICE'].sort_values(ascending=False)


# In[42]:


plt.figure(figsize = (14,13))
sns.lmplot(x = 'REG_FEE',y = 'SALES_PRICE', data = df,hue='PARK_FACIL',palette = 'coolwarm',aspect=3,height=6)


# In[43]:


df.columns


# In[44]:


plt.figure(figsize = (14,13))
sns.scatterplot(x = 'REG_FEE',y = 'SALES_PRICE', data = df)#,hue='PARK_FACIL',palette = 'coolwarm')


# In[45]:


df['DATE_BUILD'] = pd.to_datetime(df['DATE_BUILD'])
df['DATE_SALE'] = pd.to_datetime(df['DATE_SALE'])


# In[46]:


df.head()


# In[47]:


df_6_less  = df[df['DATE_SALE'] < '2006-01-01']
df_6_10 = df[(df['DATE_SALE'] < '2010-01-01') & (df['DATE_SALE'] >= '2006-01-01')]
df_10_14 = df[(df['DATE_SALE'] < '2014-01-01') & (df['DATE_SALE'] >= '2010-01-01')]
df_14_more = df[df['DATE_SALE'] >= '2014-01-01']


# In[48]:


sns.lmplot(x = 'REG_FEE',y = 'SALES_PRICE', data = df_6_less,hue='PARK_FACIL',palette = 'coolwarm',aspect=3,height=8)


# In[49]:


sns.lmplot(x = 'REG_FEE',y = 'SALES_PRICE', data = df_6_10,hue='PARK_FACIL',palette = 'coolwarm',aspect=3,height=8)


# In[50]:


sns.lmplot(x = 'REG_FEE',y = 'SALES_PRICE', data = df_10_14,hue='PARK_FACIL',palette = 'coolwarm',aspect=3,height=8)


# In[51]:


sns.lmplot(x = 'REG_FEE',y = 'SALES_PRICE', data = df_14_more,hue='PARK_FACIL',palette = 'coolwarm',aspect=3,height=8)


# In[52]:


sns.barplot(x='AREA', y='SALES_PRICE',data = df_6_less)


# In[53]:


plt.figure(figsize = (12,7))
sns.barplot(x='AREA', y='SALES_PRICE',data = df_6_10)


# In[54]:


plt.figure(figsize = (12,7))
sns.barplot(x='AREA', y='SALES_PRICE',data = df_10_14)


# In[55]:


plt.figure(figsize = (12,7))
sns.barplot(x='AREA', y='SALES_PRICE',data = df_14_more)


# In[56]:


df_annanagar = df[df['AREA'] == 'Anna Nagar']
df_tnagar = df[df['AREA']== 'T Nagar']
df_velachery = df[df['AREA'] == 'Velachery']
df_chrompet = df[df['AREA'] == 'Chrompet']
df_karapakkam = df[df['AREA'] == 'Karapakkam']
df_adyar = df[df['AREA'] == 'Adyar']
df_kknagar = df[df['AREA'] == 'KK Nagar']


# In[57]:


sns.barplot(x='N_BEDROOM',y='SALES_PRICE',data=df_annanagar,hue='PARK_FACIL')


# In[58]:


sns.barplot(x='N_BEDROOM',y='SALES_PRICE',data=df_tnagar,hue='PARK_FACIL')


# In[59]:


sns.barplot(x='N_BEDROOM',y='SALES_PRICE',data=df_kknagar,hue='PARK_FACIL')


# In[60]:


sns.barplot(x='N_BEDROOM',y='SALES_PRICE',data=df_chrompet,hue='PARK_FACIL')


# In[61]:


sns.barplot(x='N_BEDROOM',y='SALES_PRICE',data=df_karapakkam,hue='PARK_FACIL')


# In[62]:


sns.barplot(x='N_BEDROOM',y='SALES_PRICE',data=df_velachery,hue='PARK_FACIL')


# In[63]:


sns.barplot(x='N_BEDROOM',y='SALES_PRICE',data=df_adyar,hue='PARK_FACIL')


# In[64]:


df.dropna(inplace=True)


# In[65]:


df.isnull().sum()


# In[66]:


df.info()


# In[67]:


df['AGE_OF_BUILDING'] = (df['DATE_SALE'] - df['DATE_BUILD']).dt.days


# In[68]:


df.head()


# In[69]:


df.drop(['PRT_ID','DATE_SALE','DATE_BUILD'],axis=1,inplace=True)


# In[70]:


df.head()


# In[71]:


df.info()


# In[72]:


df.head()


# In[73]:


df = pd.get_dummies(df,drop_first=True)


# In[74]:


df.head()


# In[75]:


df.shape


# In[76]:


df.corr()['SALES_PRICE'].sort_values(ascending=False)

# ## Linear Regression

# In[77]:


from sklearn.linear_model import LinearRegression


# In[78]:


from sklearn.model_selection import train_test_split


# In[79]:


X1 = df.drop('SALES_PRICE',axis=1)
y1 = df['SALES_PRICE']


# In[80]:


X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size=0.3)


# In[81]:


linear_model = LinearRegression()


# In[82]:


linear_model.fit(X1_train,y1_train)


# In[83]:


pred = linear_model.predict(X1_test)


# In[84]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score


# In[85]:


print(mean_absolute_error(y1_test,pred))


# In[86]:


df.describe()['SALES_PRICE']


# In[87]:


print(mean_squared_error(y1_test,pred))


# In[88]:


print(np.sqrt(mean_squared_error(y1_test,pred)))


# In[89]:


print(explained_variance_score(y1_test,pred))


# In[245]:


accuracy = linear_model.score(X1_test,y1_test)
print(accuracy*100)                   #Accuracy of linear model


# ## ANN

# In[90]:


X = df.drop('SALES_PRICE',axis=1).values
y = df['SALES_PRICE'].values


# In[91]:


from sklearn.preprocessing import MinMaxScaler


# In[92]:


scaler = MinMaxScaler()


# In[93]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[94]:


scaler.fit(X_train)


# In[95]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[96]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[97]:


from tensorflow.keras.callbacks import EarlyStopping


# In[195]:


early_stop = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience = 40)


# In[196]:


model = Sequential()


# In[197]:


model.add(Dense(35,activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(17,activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(8,activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(4,activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(1))


# In[198]:


model.compile(optimizer = 'adam', loss='mse')


# In[199]:


model.fit(x = X_train, y = y_train, epochs = 1000, validation_data = (X_test,y_test), verbose = 10, callbacks = [early_stop])


# In[201]:


loss_df = pd.DataFrame(model.history.history)


# In[202]:


loss_df.plot()


# In[203]:


model.history.history


# In[204]:


model.evaluate(X_test,y_test)


# In[205]:


predi = model.predict(X_test)


# In[206]:


predi = pd.Series(predi.reshape(2131,))


# In[207]:


pred_df = pd.DataFrame(y_test)
pred_df = pd.concat([pred_df,predi] , axis=1)
pred_df.columns = ['Test True Y', 'Predicted Y']


# In[208]:


sns.scatterplot(x='Test True Y',y='Predicted Y',data=pred_df)


# In[209]:


print(np.sqrt(mean_squared_error(y_test,predi)))


# In[210]:


print(mean_absolute_error(y_test,predi))


# In[211]:


print(explained_variance_score(y_test,predi))       #Accuracy of neural network


# In[146]:


df.head()


# ### Inputting new values and checking the predicted values

# In[213]:


new_value = df.drop('SALES_PRICE',axis=1).iloc[0].values
new_value = new_value.reshape(-1,35)


# In[214]:


new_value_ans = df.iloc[0]['SALES_PRICE']


# In[215]:


new_value = scaler.transform(new_value)


# In[216]:


model_predicted_new_value_ans = model.predict(new_value)


# In[217]:


new_value_ans - model_predicted_new_value_ans


# In[ ]:





# In[218]:


new_value3 = df.drop('SALES_PRICE',axis=1).iloc[3].values
new_value3 = new_value3.reshape(-1,35)


# In[219]:


new_value3_ans = df.iloc[3]['SALES_PRICE']


# In[220]:


linearmodel_predicted_new_value_ans = linear_model.predict(new_value3)


# In[221]:


new_value3_ans - linearmodel_predicted_new_value_ans


# In[ ]:





# In[222]:


new_value3_scaled = scaler.transform(new_value3)


# In[223]:


model_predicted_new_value3_ans = model.predict(new_value3_scaled)


# In[224]:


new_value3_ans - model_predicted_new_value3_ans


# In[225]:


from sklearn import metrics


# In[226]:


metrics.r2_score(y1_test,pred)


# In[227]:


metrics.r2_score(y_test,predi)


# In[230]:


accuracy = linear_model.score(X1_test,y1_test)
print(accuracy*100)                   #Accuracy of linear model


# ## Random Forest Regressor

# In[232]:


from sklearn.ensemble import RandomForestRegressor


# In[239]:


regressor = RandomForestRegressor(n_estimators = 500)


# In[240]:


regressor.fit(X1_train,y1_train)


# In[241]:


pred_rfc = regressor.predict(X1_test)


# In[242]:


accuracy_rfc = regressor.score(X1_test,y1_test)
print(accuracy_rfc*100)      


# In[243]:


print(explained_variance_score(y1_test,pred_rfc))


# In[244]:


accuracy_rfc = regressor.score(X1_train,y1_train)
print(accuracy_rfc*100)  #Training data


# In[ ]:




