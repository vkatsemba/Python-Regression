##This code test the logistic regression with the NESARC Survey data

##Import the modules
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

##Read the data set and select the columns for the response and explanatory variables
data = pd.read_csv('C:\\Users\\Vadim Katsemba\\Documents\\nesarc_pds.csv', usecols=['S1Q7A9','AGE','CONSUMER','NUMPERS','ETHRACE2A'])

##Customize the data frame and remove missing and unknown values
dataf = pd.DataFrame()

dataf['RETIRED'] = data['S1Q7A9'].replace(' ',np.NaN).replace('2','0').astype(float)

dataf['AGE'] = data['AGE'].replace(' ',np.NaN).replace('98',np.NaN).astype(float)
dataf['DRINKSTATUS'] = data['CONSUMER'].replace(' ',np.NaN).replace('999',np.NaN).astype(float)
dataf['HOUSE_PEOPLE'] = data['NUMPERS'].replace(' ',np.NaN).astype(float)
dataf['RACE'] = data['ETHRACE2A'].replace(' ',np.NaN)

##Center the means
for c in ['AGE','DRINKSTATUS','HOUSE_PEOPLE']:
    dataf[c] = dataf[c]-dataf[c].mean()

dataf[['AGE','DRINKSTATUS','HOUSE_PEOPLE']].describe()

##Run the logistic regression with Race as a categorical variable
logm = smf.logit(formula='RETIRED ~ AGE + DRINKSTATUS + HOUSE_PEOPLE + C(RACE)',data=dataf).fit()
print(logm.summary())

##Generate a table of confidence intervals, odd ratios and p-values
conf = logm.conf_int()
conf.columns = ['Lower CI','Upper CI']
conf['OR'] = logm.params
conf = np.exp(conf)
conf['p-val'] = logm.pvalues.round(3)
print(conf)
