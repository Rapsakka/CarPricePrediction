import os
import math
import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error




def main():
    Car_data = pd.read_csv('car_price_prediction.csv')

    # Car_data.shape
    # 19237 rows, 18 colums

    # Car_data.info()

    # Car_data.describe()

    # data preprocessing -need to clear gearbox type,

    data = Car_data.drop(['ID', 'Levy', 'Wheel','Color','Category'], axis=1)

    # remove other tarnsmissions besides Automatic or manual, and categorizise transmissions
    # 0 = manual
    # 1 = automatic
    data = data.loc[data['Gear box type'].isin(['Automatic', 'Manual'])]
    data.loc[data['Gear box type'] == 'Manual', 'Gear box type'] = 0
    data.loc[data['Gear box type'] == 'Automatic', 'Gear box type'] = 1

    # change door number to integer
    # the data seems to have weird feture where some cars have '>5' marked for doors.This seems to be caused by trunck interpeted as a door

    data.loc[data['Doors'] == '04-May', 'Doors'] = 4
    data.loc[data['Doors'] == '02-Mar', 'Doors'] = 2
    data.loc[data['Doors'] == '>5', 'Doors'] = 4

    # change car prod.year to car age(from this year)
    data['Age'] = 2022 - data['Prod. year']
    data.drop('Prod. year', axis=1, inplace=True)

    # change the fuel type to either 0, 1 or 2
    # 0 = Petrol
    # 1 = diesel
    # 2 = hybrid
    # CNG values are deleted as CNG is marginal in finland and many other countries
    # same for LPG
    data.loc[data['Fuel type'] == 'Petrol', 'Fuel type'] = 0
    data.loc[data['Fuel type'] == 'Diesel', 'Fuel type'] = 1
    data.loc[data['Fuel type'] == 'Hybrid', 'Fuel type'] = 2
    data.loc[data['Fuel type'] == 'Plug-in Hybrid', 'Fuel type'] = 2
    data = data[data['Fuel type'] != 'CNG']
    data = data[data['Fuel type'] != 'LPG']

    # change the Drive wheels to either 0, 1 or 2
    # 0 = Front
    # 1 = Back
    # 2 = 4x4
    data.loc[data['Drive wheels'] == 'Front', 'Drive wheels'] = 0
    data.loc[data['Drive wheels'] == 'Rear', 'Drive wheels'] = 1
    data.loc[data['Drive wheels'] == '4x4', 'Drive wheels'] = 2

    # change 'Leather interior' to binary
    data.loc[data['Leather interior'] == 'Yes', 'Leather interior'] = 1
    data.loc[data['Leather interior'] == 'No', 'Leather interior'] = 0

    # mileage needs to be converted to int and km from the end needs to be removed
    def remove_km(value):
        km = str(value)
        final = int(km[:-3])
        return final

    data['Mileage'] = data['Mileage'].apply(remove_km)

    def to_float(a):
        if len(a) > 3:
            b = a[:3]
        else:
            b = a
        return float(b)

    data['Engine volume'] = data['Engine volume'].apply(to_float)

    # one price is super high this could be considered as an outlier e.g. vintage car and us such should be removed as the model is for "everyday cars"
    data = data[data.Price < 1e7]
    # sns.boxplot(x='Price',data = data)
    # sns.boxplot(x='Mileage',data = data)
    # data.head()
    # data.info()

    manu = Car_data.Manufacturer.unique().tolist()
    manu_val = zip( list( range( 0,len( Car_data.Manufacturer.unique() ) ) ), manu)

    for (value,manufactur) in manu_val:
        data.loc[ data['Manufacturer'] == manufactur, 'Manufacturer' ] = value

    # for the memes
    mode = Car_data.Model.unique().tolist()
    mode_val = zip( list( range(0, len( Car_data.Model.unique().tolist() ) ) ), mode )

    for (value, mod) in mode_val:
        data.loc[data['Model'] == mod, 'Model'] = value

    # data = Car_data.drop([ 'Color'], axis=1)
    """color = Car_data.Color.unique().tolist()
    color_val = zip(list(range(0, len(Car_data.Color.unique().tolist()))), color)

    for (value, col) in color_val:
        data.loc[data['Color'] == col, 'Color'] = value
   
    category = Car_data.Category.unique().tolist()
    category_val = zip(list(range(0, len(Car_data.Category.unique().tolist()))), category)

    for (value, cate) in category_val:
        data.loc[data['Category'] == cate, 'Category'] = value
     """

    X = data.drop(['Price'], axis=1).to_numpy()
    y = data['Price'].to_numpy()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    regr_1 = tree.DecisionTreeRegressor(max_depth=2)
    regr_2 = tree.DecisionTreeRegressor(max_depth=8)
    regr_1.fit(X_train, y_train)
    regr_2.fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=500, max_features='sqrt', max_depth=10, random_state=18).fit(X_train,
                                                                                                        y_train)
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    y_3 = rf.predict(X_test)

    #error1 = mean_squared_error(y_test,y_1)
    #print( math.sqrt( error1 ))
    #error2 = mean_squared_error(y_test, y_2)
    #print( math.sqrt( error2 ) )
    error3 = mean_squared_error(y_test, y_3)
    print( math.sqrt( error3 ) )

    #print( data.info() )

if __name__ == "__main__":
    main()