##This code uses the MarsCrater data set to test a Basic Linear Regression model

##Import modules
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns

##Read the data file and select the Diameter and Latitude columns
data = pd.read_csv('C:\\Users\\Vadim Katsemba\\Documents\\marscrater_pds.csv', usecols=['DIAM_CIRCLE_IMAGE','LATITUDE_CIRCLE_IMAGE'])

##Customize the data frame and remove unknown or missing values from the selected columns
dataf = pd.DataFrame()
dataf['DIAMETER'] = data['DIAM_CIRCLE_IMAGE'].replace(' ',np.NaN).replace('99',np.NaN).astype(float)
dataf['LATITUDE'] = data['LATITUDE_CIRCLE_IMAGE'].replace(' ',np.NaN).astype(float)
dataf = dataf.dropna()

##Center the means
print('Original')
print(dataf.describe())
dataf['DIAMETER'] = dataf['DIAMETER']- dataf['DIAMETER'].mean()
print('\n\nCentered DIAMETER')
print(dataf.describe())

##Run the OLS regression and view the summary
lm = smf.ols(formula='LATITUDE~DIAMETER',data=dataf).fit()
print lm.summary()

##Generate a scatterplot and fit the OLS line
sns.lmplot(x='DIAMETER', y='LATITUDE', data=dataf)
