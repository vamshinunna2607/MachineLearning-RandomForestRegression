import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn
from pandas import ExcelWriter
dataset = pd.read_csv('dataset.csv')





dataset = dataset.fillna(value=dataset.mean())
(dataset['BsmtFinType2'].fillna(value='VARIOUS',inplace=True))
(dataset['BsmtFinType1'].fillna(value='VARIOUS',inplace=True))
(dataset['BsmtExposure'].fillna(value='VARIOUS',inplace=True))
(dataset['MasVnrType'].fillna(value='VARIOUS',inplace=True))
(dataset['Alley'].fillna(value='VARIOUS',inplace=True))
(dataset['FireplaceQu'].fillna(value='VARIOUS',inplace=True))
(dataset['GarageType'].fillna(value='VARIOUS',inplace=True))
(dataset['GarageFinish'].fillna(value='VARIOUS',inplace=True))
(dataset['GarageCond'].fillna(value='VARIOUS',inplace=True))
(dataset['Fence'].fillna(value='VARIOUS',inplace=True))

(dataset['GarageQual'].fillna(value='VARIOUS',inplace=True))
(dataset['PoolQC'].fillna(value='VARIOUS',inplace=True))
(dataset['MiscFeature'].fillna(value='VARIOUS',inplace=True))

dataset = pd.get_dummies(dataset,columns=['Street','Alley','FireplaceQu','GarageType',
'GarageFinish','GarageCond','Fence','BsmtQual','BsmtCond',
'LotShape','LandContour','LotConfig','LandSlope',
'Condition1','BldgType','RoofStyle',
'MasVnrType','ExterQual',
'ExterCond','Foundation','BsmtExposure',
'BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir',
'PavedDrive','SaleCondition','GarageQual','PoolQC','MiscFeature',
'GarageQual','PoolQC','MiscFeature','Condition2','HouseStyle','RoofMatl',
'Exterior1st','Heating','Electrical','Neighborhood',
'Exterior2nd','KitchenQual',
'Functional','SaleType','MSZoning','Utilities'],drop_first=True)

"""
dataclean = dataset.dropna()
print(scipy.stats.pearsonr(dataclean['MoSold'],dataclean['YrSold']))
"""


#(dataset.columns.get_loc("SalePrice"))
#((dataset.isnull().sum()))
y = dataset.iloc[:,37].values          
X = dataset.drop(['SalePrice'],axis=1,inplace=True)



import statsmodels.formula.api as sm
dataset=np.append(arr= np.ones((1460,1)).astype(int), values = dataset,axis=1)
def backwardElimination(dataset, sl):
    numVars = len(dataset[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, dataset).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    dataset = np.delete(dataset, j, 1)
    print(regressor_OLS.summary())
    return dataset


""" 

import statsmodels.formula.api as sm
dataset=np.append(arr= np.ones((1460,1)).astype(int), values = dataset,axis=1)
def backwardElimination(dataset, SL):
    numVars = len(dataset[0])
    temp = np.zeros((1460,1)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, dataset).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = dataset[:, j]
                    dataset = np.delete(dataset, j, 1)
                    tmp_regressor = sm.OLS(y, dataset).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        dataset_rollback = np.hstack(dataset, temp[:,[0,j]])
                        dataset_rollback = np.delete(dataset_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return dataset_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return dataset

"""


SL = 0.05
X_opt = dataset[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,
78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,
112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,
140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,
168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,
196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,
224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,
252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270]]
dataset_Modeled = backwardElimination(X_opt, SL)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset_Modeled, y, test_size = 0.2,random_state = 0)




from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)



 



from sklearn.model_selection import cross_val_score
rsquare = cross_val_score(estimator = regressor,
                       X = X_train,
                       y = y_train,
                       scoring = 'r2',
                       cv = 10)

#neg_mean_absolute_error
#neg_mean_squared_error
#r2










#part5555
#(dataset.isnull().sum())
             
#part1