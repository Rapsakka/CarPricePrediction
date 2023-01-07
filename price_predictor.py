import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error


Car_data_audi = pd.read_csv('audi.csv')

Car_data_audi['manufacturer'] = 'Audi'

Car_data_bmw = pd.read_csv('bmw.csv')

Car_data_bmw['manufacturer'] = 'BMW'

Car_data_merc_cclas = pd.read_csv('cclass.csv')
Car_data_merc_cclas['manufacturer'] = 'Mercedes'

Car_data_merc = pd.read_csv('merc.csv')

Car_data_merc['manufacturer'] = 'Mercedes'

Car_data_ford_focus = pd.read_csv('focus.csv')

Car_data_ford_focus['manufacturer'] = 'Ford'

Car_data_ford = pd.read_csv('ford.csv')
Car_data_ford['manufacturer'] = 'Ford'

Car_data_hyuindi = pd.read_csv('hyundi.csv')
Car_data_hyuindi['manufacturer'] = 'Hyuindi'

Car_data_skoda = pd.read_csv('skoda.csv')
Car_data_skoda['manufacturer'] = 'Skoda'

Car_data_toyota = pd.read_csv('toyota.csv')
Car_data_toyota['manufacturer'] = 'Toyota'

Car_data_vw = pd.read_csv('vw.csv')
Car_data_vw['manufacturer'] = 'VW'

data = pd.concat([Car_data_audi, Car_data_bmw, Car_data_merc, Car_data_merc_cclas, Car_data_ford_focus, Car_data_ford,
                  Car_data_hyuindi, Car_data_skoda, Car_data_vw])

plt.figure()
plt.title("Car listing price box plot")
box = sns.boxplot(x='price', data=data)
median = data['price'].median()

print("median of car prices " + str( median) )
plt.show()

data = data.drop(['tax'], axis=1)
data = data.drop(['tax(£)'], axis=1)


# change the fuel type to either 0, 1 or 2
# 0 = Petrol
# 1 = diesel
# 2 = hybrid
# 3 = electric
# Rows where other is a value are discarded
data.loc[data['fuelType'] == 'Petrol', 'fuelType'] = 0
data.loc[data['fuelType'] == 'Diesel', 'fuelType'] = 1
data.loc[data['fuelType'] == 'Hybrid', 'fuelType'] = 2
data.loc[data['fuelType'] == 'Electric', 'fuelType'] = 3
data = data[data['fuelType'] != 'Other']

data = data.loc[data['transmission'].isin(['Automatic', 'Manual'])]
data.loc[data['transmission'] == 'Manual', 'transmission'] = 0
data.loc[data['transmission'] == 'Automatic', 'transmission'] = 1

manu = data.manufacturer.unique().tolist()
manu_val = zip(list(range(0, len(data.manufacturer.unique()))), manu)

for (value, manufactur) in manu_val:
    data.loc[data['manufacturer'] == manufactur, 'manufacturer'] = value

mode = data.model.unique().tolist()
mode_val = zip(list(range(0, len(data.model.unique().tolist()))), mode)

for (value, mod) in mode_val:
    data.loc[data['model'] == mod, 'model'] = value

data = data.dropna()

data['Age'] = 2022 - data['year']
data.drop('year', axis=1, inplace=True)

X = data.drop(['price'], axis=1).to_numpy()
y = data['price'].to_numpy()

plt.figure(figsize=(10,10))
correlations = data[data.columns].corr(method='pearson')
sns.heatmap(correlations, annot = True)
plt.show()

#from sklearn.model_selection import KFold
#k, shuffle, seed = 5, True, 42
#kfold = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
#tr_errors1 = {}
#val_errors1 = {}
#tr_errors2 = {}
#val_errors2 = {}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestRegressor(n_estimators=500, criterion='absolute_error', max_features='sqrt', max_depth=10, random_state=18).fit(X_train,
                                                                                                     y_train)
y_pred1 = rf.predict(X_test)
print("tree done")

mplregr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
y_pred2 = mplregr.predict(X_test)

train_error1 = mean_absolute_error(y_train,y_pred1)
test_error1 = mean_absolute_error(y_test, y_pred1)

train_error2 = mean_absolute_error(y_train,y_pred2)
test_error2 = mean_absolute_error(y_test, y_pred2)
mplregr.score(X_test, y_test)


print("Random forest absolute TRAIN error : " + str(train_error1))
print("Random forest absolute error : " + str(test_error1))
print("Random forest score(best score is 1.0): " + str(rf.score(X_test, y_test)))
print()
print()
print("Multi-layer perceptron absolute TRAIN error: " + str(train_error2))
print("Multi-layer perceptron absolute error: " + str(test_error2))
print("Multi-layer perceptron score(best score is 1.0): " + str(mplregr.score(X_test, y_test)))
print()
print()

data.info()
