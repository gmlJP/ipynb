
# coding: utf-8

# In[2]:

import pandas as pd
import os
dir_path= "/vagrant/titanic/"
df = pd.read_csv(os.path.join(dir_path, 'train.csv'), header=0)

# [              1st 2nd 3rd 
#  female ->  0  [0,   1,  2 , 
#  male   ->  1  [0,   1,  2]
# ]

median_ages = np.zeros((2,3)) # 2x3

df['Gender'] = df.Sex.map( {'female': 0, 'male' : 1} ).astype(int)

for i in range(0,2): # [0, 1]
    for j in range(0,3): # [0,1,2]
        median_ages[i,j] = df[(df.Gender == i) & (df.Pclass == j+1)]['Age'].dropna().median()
        
# Make a copy of Age
df['AgeFill'] = df['Age']

for i in range(0,2): # [0, 1]
    for j in range(0,3): # [0,1,2]
        df.loc[ (df.AgeFill.isnull()) & (df.Gender == i) & (df.Pclass == j+1) , 'AgeFill'] = median_ages[i,j]

df['FamilySize'] = df['SibSp'] + df['Parch']

df['Age*Class'] = df.AgeFill * df.Pclass

print df.dtypes[df.dtypes.map(lambda x: x=='object')]

df = df.drop(['Age'], axis=1)

df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

print
print "df.describe()"
print df.describe()
print
print
print "df.info()"
print df.info()
print

train_data = df.values
print
print "train_data" 
print train_data
print


# In[3]:

train_data[0::, 1] # classes


# In[4]:

train_data[0]


# In[5]:

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)


# In[6]:

df = pd.read_csv(os.path.join(dir_path, 'test.csv'), header=0)
test_data = df.values
print test_data

#forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

#output = forest.predict(test_data)


# In[7]:

zip((2,3,4,5,9), (10,11,23,85,100))


# In[8]:

from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
data.shape


# In[9]:

data


# In[10]:

digits = datasets.load_digits()
digits.images.shape


# In[12]:

import pylab as pl
pl.imshow(digits.images[-1], cmap=pl.cm.Greens_r) 


# In[14]:

data = digits.images.reshape((digits.images.shape[0], -1))
pl.imshow(digits.images[-1], cmap=pl.cm.gray_r) 

