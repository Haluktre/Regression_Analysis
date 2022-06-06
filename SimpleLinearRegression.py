import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Data/satislar.csv")

print(df)

aylar = df[["Aylar"]]
print(aylar)

satislar = df[["Satislar"]]
print(satislar)

#train ve test olarak bölme
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

#model inşası basit doğrusal regresyon
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

#Verileri sıralama ve görselleştirme

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train,"o")
plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")









