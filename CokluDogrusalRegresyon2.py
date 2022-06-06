import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Data/odev_tenis.csv")

print(df)

from sklearn import preprocessing

outlook = df.iloc[:,0:1].values
le = preprocessing.LabelEncoder()

outlook[:,0] = le.fit_transform(df.iloc[:,0])

print(outlook)

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

outlook = pd.DataFrame(data=outlook, columns = ["overcast","rainy","sunny"])
print(outlook)
df = pd.concat([df,outlook],axis=1)
df = df.drop(["outlook"],axis=1)
print(df)


windy = df["windy"].values
le = preprocessing.LabelEncoder()
windy = le.fit_transform(df["windy"])

df["windy"] = windy

play = df["play"].values
le = preprocessing.LabelEncoder()
play = le.fit_transform(df["play"])

df["play"] = play

print(df)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop(["humidity"],axis=1),df["humidity"],test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#Model anlamlılık değerleri
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int),values=df,axis=1)

X_l = df.drop(["humidity"],axis=1).values
X_l = np.array(X_l,dtype=float)

model = sm.OLS(df["humidity"],X_l).fit()

print(model.summary())











