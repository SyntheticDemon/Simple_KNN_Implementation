
from hashlib import new
import numpy as np
from numpy.lib.function_base import diff
from numpy.lib.index_tricks import index_exp
from numpy.lib.npyio import recfromcsv
import pandas as pd
def train():
    df = pd.read_csv('train.csv')
    test_data= clean_data(df)
    df=df.to_numpy()
    for k in range(len(test_data)):
        corrent_count =0
        for i in range(len(test_data)):
               if(knn(test_data,df,i,k)==df[int(test_data.to_numpy()[i][0]-1)][1]):
                   corrent_count+=1
        print("efficacy for k =" + f"{k} is " +str(corrent_count/len(test_data)))
def knn(test_data,real_data,test_id,default_k=3):
    new_list = test_data.to_numpy()
    element_distance_list=[]
    for i in range(len(new_list)):
        if i!=test_id:
            element_distance_list.append((get_difference(new_list[i],new_list[test_id]),int(new_list[i][0]-1)))
    element_distance_list.sort(key= lambda x:x[0])
    survival_count = 0
    for i in range(default_k):
        survival_count+=real_data[element_distance_list[i][1]][1]
    if (survival_count>default_k-survival_count):
        return 1
    else:
        return 0
def determine(a,list):
    if a==np.NaN or a==np.nan:
        return list.mode()
    else:
        return a
def get_difference(vector_a,vector_b):#Euclidean Distance used(With no predesposition)
   sum=0
   for i in range(1,len(vector_a)):
        if (i==2 or i==3):
            sum+=((vector_b[i]- vector_a[i])*20  )**2
        else :
            sum+=((vector_b[i]- vector_a[i]) )**2
   difference=np.sqrt(sum)
   return difference
def clean_data(data_set):
    gender={"male":0,"female":1}
    reform=data_set.drop(['Cabin','Name'],axis=1) 
    reform=reform.drop(['Embarked'],axis=1)
    reform=reform.drop(['Survived'],axis=1)
    reform["Sex"]=reform["Sex"].apply(lambda a:gender[a]) #Replaced some missing values with the mode , some with the average , some are totally dropped
    reform=reform.drop(['Ticket'],axis=1) 
    reform["Age"]=reform["Age"].apply(lambda a :determine(a,reform["Age"]))
    passenger_class_average=np.average(data_set['Pclass'],axis=None,weights=None)
    passenger_class_average=np.round(passenger_class_average)
    print(reform)
    reform=reform.dropna()
    return reform
train()