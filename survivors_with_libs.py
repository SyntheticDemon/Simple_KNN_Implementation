
from hashlib import new
from survivors import train
import numpy as np
from numpy.lib.function_base import diff
from numpy.lib.index_tricks import index_exp
from numpy.lib.npyio import recfromcsv
from sklearn.neighbors import KNeighborsClassifier
import scipy as sp
import pandas as pd
def determine(a,list):
    if a==np.NaN or a==np.nan:
        return list.mode()
    else:
        return a
def clean_data(data_set):
    embarked_dic ={"S":1,"C":2,"Q":3,np.NaN:0}    
    gender={"male":0,"female":1}
    reform=data_set.drop(['Survived'],axis=1) 
    reform["Embarked"]=reform["Embarked"].apply((lambda a:embarked_dic[a]))
    reform["Sex"]=reform["Sex"].apply(lambda a:gender[a]) #Replaced some missing values with the mode , some with the average , some are totally dropped
    reform=reform.drop(['Ticket'],axis=1) 
    reform= reform.drop(['Name'],axis=1) 
    reform=reform.drop(['Cabin'],axis=1)
    reform=reform.drop(['Fare'],axis=1)
    reform["Age"]=reform["Age"].apply(lambda a :determine(a,reform["Age"]))
    passenger_class_average=np.average(data_set['Pclass'],axis=None,weights=None)
    passenger_class_average=np.round(passenger_class_average)
    reform=reform.dropna()
    return reform
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
test_data= clean_data(train_df)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

