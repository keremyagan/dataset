import pandas as pd
import numpy as np

veriler=pd.read_csv("https://github.com/keremyagan/dataset/raw/main/water_potability.csv")

veriler=veriler.drop('Sulfate',axis=1)
veriler=veriler.dropna(how="any")

x=veriler.iloc[:,[1,2,3,6,8]]
y=veriler.iloc[:,0:1]

X=x.values
Y=y.values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=15)

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
sc.fit(x_train)
x_test=sc.transform(x_test)
x_train=sc.transform(x_train)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping

model=Sequential()

model.add(Dense(units=30,activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=30,activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=30,activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=1))

model.compile(loss="mse",optimizer="adam")

earlyStopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)

model.fit(x=x_train,y=y_train,epochs=700,validation_data=(x_test,y_test),verbose=1,callbacks=[earlyStopping])

#tahminleri dataframe'e eklemek
tahminler=model.predict(x_test)
dizi1=pd.DataFrame(tahminler,columns=["Tahmin Degerler"])
dizi2=pd.DataFrame(y_test,columns=["Gercek Degerler"])
dizi3=pd.concat([dizi2,dizi1],axis=1)

#model doğruluk payını değerlendirmek
from sklearn.metrics import mean_absolute_error,mean_squared_error
absolute_error=mean_absolute_error(dizi3["Gercek Degerler"],dizi3["Tahmin Degerler"])
print(absolute_error)#ortalama sapmayı gösterir
