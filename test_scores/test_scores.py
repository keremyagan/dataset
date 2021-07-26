import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import statsmodels.api as sm

veriler=pd.read_csv('https://github.com/keremyagan/dataset/raw/main/test_scores/test_scores.csv')

"""
student_id=gereksiz çünkü satır numarası da eşsiz(unique) değer
classroom= dağılımda eşitsizlik var, %1 %97 gibi
school = dağılımda eşitsizlik var %7 %86 gibi
"""
veriler=veriler.drop(["student_id","classroom","school"],axis=1) 

#string veriler numerik hale getirilir
from sklearn import preprocessing
veriler=pd.concat([veriler.iloc[:,[0,1,2,4,5]].apply(preprocessing.LabelEncoder().fit_transform),veriler.iloc[:,[3,6,7]]],axis=1) 

"""
veriler.isnull().sum() #eksik veri var mı diye kontrol edilir
veriler.describe()    #veri hakkında bilgi edinilir
veriler.corr()["posttest"].sort_values()  #test kümesinin diğer kümeler ile ilişkisine bakılır(azdan çoğa sıralı şekilde)
veriler.boxplot() #veri aralığı incelenir
sns.distplot(veriler.iloc[:,[5]]) #veri dağılımına bakılır
"""

x=veriler.iloc[:,[0,1,2,3,4,5,6]]
y=veriler.iloc[:,7:8] 

X=x.values
Y=y.values

tahmin_verisi=[[2,0,1,0,0,20,66]] #79

#veriler test ve train olarak ayrılır
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=44)

#çoklu linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)

model=sm.OLS(lin_reg.predict(x_test),y_test).fit()
#print(model.summary())

r2=r2_score(y_test,lin_reg.predict(x_test))
absolute_error=mean_absolute_error(y_test,lin_reg.predict(x_test))

print("Linear Regresyon")
print(f"R2 Değeri:{r2}")
print(f"Tahmin Sonucu:{lin_reg.predict(tahmin_verisi)}")
print(f"Absolute Error:{absolute_error} ")

#polinomik regresyon
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4) 
x_poly=poly_reg.fit_transform(x_train) #veri polinom hale getirildi
lin_reg.fit(x_poly,y_train)

model2=sm.OLS(lin_reg.predict(poly_reg.fit_transform(x_test)),y_test).fit()
#print(model2.summary())

r2=r2_score(y_test,lin_reg.predict(poly_reg.fit_transform(x_test)))
absolute_error=mean_absolute_error(y_test,lin_reg.predict(poly_reg.fit_transform(x_test)))

print("Polinomik Regresyon")
print(f"R2 Değeri:{r2}")
print(f"Tahmin Sonucu:{lin_reg.predict(poly_reg.fit_transform(tahmin_verisi))}")
print(f"Absolute Error:{absolute_error} ")

#veri yakınlaştırılıyor
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(x_train)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(y_train)
sc3=StandardScaler()
x_test_olcekli=sc3.fit_transform(x_test)
sc4=StandardScaler()
y_test_olcekli=sc4.fit_transform(y_test)
#svr kullanılıyor
from sklearn.svm import SVR
svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

model3=sm.OLS(svr_reg.predict(x_test_olcekli),y_test_olcekli).fit()
#print(model3.summary())

r2=r2_score(y_test_olcekli,svr_reg.predict(x_test_olcekli))
absolute_error=mean_absolute_error(y_test_olcekli,svr_reg.predict(x_test_olcekli))

print("SVR")
print(f"R2 Değeri:{r2}")
print(f"Tahmin Sonucu:{(svr_reg.predict(tahmin_verisi))}")
print(f"Absolute Error:{absolute_error} ")

#random forest kullanma
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=20,random_state=0)
rf_reg.fit(x_train,y_train)

model5=sm.OLS(rf_reg.predict(x_test),y_test).fit()
#print(model5.summary())

r2=r2_score(y_test,rf_reg.predict(x_test))
absolute_error=mean_absolute_error(y_test,rf_reg.predict(x_test))

print("Random Forest")
print(f"R2 Değeri:{r2}")
print(f"Tahmin Sonucu:{rf_reg.predict(tahmin_verisi)}")
print(f"Absolute Error:{absolute_error} ")

#Tensforflow
x=veriler.iloc[:,[1,2,3,4,5,6]]
y=veriler.iloc[:,7:8] 

X=x.values
Y=y.values

tahmin_verisi=[[0,1,0,0,20,66]] #79

#veriler test ve train olarak ayrılır
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=44)


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

model.add(Dense(units=150,activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=150,activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=150,activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=1))

model.compile(loss="mse",optimizer="adam")

earlyStopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)

model.fit(x=x_train,y=y_train,epochs=700,validation_data=(x_test,y_test),verbose=1,callbacks=[earlyStopping])

tahminler=model.predict(x_test)
dizi1=pd.DataFrame(tahminler,columns=["Tahmin Degerler"])
dizi2=pd.DataFrame(y_test,columns=["Gercek Degerler"])
dizi3=pd.concat([dizi2,dizi1],axis=1)

#model doğruluk payını değerlendirmek
absolute_error=mean_absolute_error(dizi3["Gercek Degerler"],dizi3["Tahmin Degerler"])
squared_error=mean_squared_error(dizi3["Gercek Degerler"],dizi3["Tahmin Degerler"])
r2=r2_score(dizi3["Gercek Degerler"],dizi3["Tahmin Degerler"])
print(absolute_error)
print("Tensorflow")
print(f"R2 Degeri:{r2}")
print(f"Absolute Error:{absolute_error} ")
tahmin_verisi=sc.transform(tahmin_verisi)
tahmin_sonucu=model.predict(tahmin_verisi)
print(f"Tahmin Sonucu:{tahmin_sonucu}")
