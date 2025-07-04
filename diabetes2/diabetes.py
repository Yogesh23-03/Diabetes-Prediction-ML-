# -*- coding: utf-8 -*-
"""diabetes.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JNjhmrLtVM6yJibvZWMNDUWavOqKa2qL
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

diabetesdata=pd.read_csv('diabetes.csv')

diabetesdata.head()

diabetesdata.isnull().sum()

diabetesdata.info()

diabetesdata.describe()

diabetesdata.groupby('Outcome').mean()

a=diabetesdata.drop(columns='Outcome',axis=1)
b=diabetesdata["Outcome"]

sl=StandardScaler()
stad=sl.fit_transform(a)

a=stad

a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.2,stratify=b,random_state=2)

classification=svm.SVC(kernel='linear')
new=classification.fit(a_train,b_train)
new

apred=classification.predict(a_train)
apred

train_accuracy=accuracy_score(apred,b_train)
train_accuracy

xpred=classification.predict(a_test)


test_accuracy=accuracy_score(xpred,b_test)
test_accuracy

input_data=(10,168,74,0,0,38,0.537,34)
input_dataasnpar=np.asarray(input_data)
datareshape=input_dataasnpar.reshape(1,-1)
std_data=sl.transform(datareshape)

pred=classification.predict(std_data)
print(pred)



import pickle
filename = 'diabetes_model.sav'
pickle.dump(classification, open(filename, 'wb'))