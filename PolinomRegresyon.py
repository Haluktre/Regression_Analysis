import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn


df = pd.read_csv("Data/BNB-USD.csv")



print(df)
plt.plot(df["Volume"], df["Close"], 'o')

plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop(["Close","Date","Open","High","Low","Adj Close"],axis=1),df["Close"],test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#Model anlamlılık değerleri
import statsmodels.api as sm

X = np.append(arr = np.ones((1660,1)).astype(int),values=df,axis=1)

X_l = df.drop(["Close","Date","Open","High","Low","Adj Close"],axis=1).values
X_l = np.array(X_l,dtype=float)

model = sm.OLS(df["Close"],X_l).fit()

print(model.summary())