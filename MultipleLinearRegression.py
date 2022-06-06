import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Data/veriler.csv')

print(df)

#Kategorik verileri sayısal veriye dönüştürme:
from sklearn import preprocessing

cinsiyet = df.iloc[:,-1:].values
print(cinsiyet)
le = preprocessing.LabelEncoder()

cinsiyet[:,-1] = le.fit_transform(df.iloc[:,-1])

print(cinsiyet)

ohe = preprocessing.OneHotEncoder()
cinsiyet = ohe.fit_transform(cinsiyet).toarray()

df["cinsiyet"] = cinsiyet

from sklearn import preprocessing

ulke = df.iloc[:,0:1].values
le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(df.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

#Verilerin birleştirilmesi

ulke = pd.DataFrame(data=ulke, columns = ["fr","tr","us"])

df = pd.concat([df,ulke],axis=1)

df = df.drop("ulke",axis=1)

print(df)

#Verileri train ve test olarak ayırma
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop(["boy","yas","cinsiyet"],axis=1),df["boy"],test_size=0.33,random_state=0)


"""from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)"""

#model inşası çoklu doğrusal regresyon
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


#Model anlamlılık değerleri
import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int),values=df,axis=1)


X_l = df.drop(["boy","yas","cinsiyet"],axis=1).values
X_l = np.array(X_l,dtype=float)

model = sm.OLS(df["boy"],X_l).fit()

print(model.summary())












