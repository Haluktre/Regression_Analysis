import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataFrame = pd.read_csv("veriler.csv")

#print(dataFrame)

#Eksik veriler için:
from sklearn.impute import SimpleImputer
    
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    
yas = dataFrame.iloc[:,1:4].values
    
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])

#print(yas)

#Kategorik verileri sayısal veriye dönüştürme:
from sklearn import preprocessing

ulke = dataFrame.iloc[:,0:1].values
print(ulke)
le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(dataFrame.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

print(ulke)

#Verilerin birleştirilmesi

sonuc = pd.DataFrame(data=ulke,index=range(22), columns = ["fr","tr","us"])
print(sonuc)

sonuc2 = pd.DataFrame(data = yas, index= range(22), columns=["boy","kilo","yas"])
print(sonuc2)

cinsiyet = dataFrame.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#Verileri train ve test olarak ayırma

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#Öznitelik ölçeklendirme

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
















