import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn
data = pd.read_csv('dataset.csv')


data = data.fillna(value=data.mean())
(data['BsmtFinType2'].fillna(value='VARIOUS',inplace=True))
(data['BsmtFinType1'].fillna(value='VARIOUS',inplace=True))
(data['BsmtExposure'].fillna(value='VARIOUS',inplace=True))
(data['MasVnrType'].fillna(value='VARIOUS',inplace=True))
(data['Alley'].fillna(value='VARIOUS',inplace=True))
(data['FireplaceQu'].fillna(value='VARIOUS',inplace=True))
(data['GarageType'].fillna(value='VARIOUS',inplace=True))
(data['GarageFinish'].fillna(value='VARIOUS',inplace=True))
(data['GarageCond'].fillna(value='VARIOUS',inplace=True))
(data['Fence'].fillna(value='VARIOUS',inplace=True))

(data['GarageQual'].fillna(value='VARIOUS',inplace=True))
(data['PoolQC'].fillna(value='VARIOUS',inplace=True))
(data['MiscFeature'].fillna(value='VARIOUS',inplace=True))


dataclean = data.dropna()
print(scipy.stats.pearsonr(dataclean['SalePrice'],dataclean['GrLivArea']))
scat1= seaborn.regplot(x="SalePrice",y="GrLivArea",fit_reg=True, data=data)
plt.xlabel('SalePrice')
plt.ylabel('GrLivArea')
plt.show()