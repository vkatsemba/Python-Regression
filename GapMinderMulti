##This code tests multiple regression with the GapMinder data set

##import modules
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

##Read data
data = pd.read_csv('C:\\Users\\Vadim Katsemba\\Documents\\gapminder.csv')

##Customize the data frame and remove missing and unknown values for the columns pertaining to the response and explanatory variables.
dataf = pd.DataFrame()

dataf['Income'] = data['incomeperperson'].replace(' ',np.NaN).astype(float)

dataf['EmployRate'] = data['employrate'].replace(' ',np.NaN).astype(float)
dataf['LifeExpect'] = data['lifeexpectancy'].replace(' ',np.NaN).astype(float)
dataf['ResidElectric'] = data['relectricperperson'].replace(' ',np.NaN).astype(float)
dataf['Urban'] = data['urbanrate'].replace(' ',np.NaN).astype(float)

##Summarize the data frame
dataf = dataf.dropna()
dataf.describe()

##Center the means
COLS = ['EmployRate','LifeExpect','ResidElectric','Urban']
for c in COLS:
    dataf[c] = dataf[c]-dataf[c].mean()
    
dataf.describe()

##Run the multiple regression
multlm = smf.ols('Income ~ EmployRate + LifeExpect + ResidElectric + Urban', data = dataf).fit()
print(multlm.summary())

##Generate the QQ Plot
qq_plot = sm.qqplot(multlm.resid, line='r')

##Generate the standardized residual plot
stdres=pd.DataFrame(multlm.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')

##Generate variable interaction plots
for c in COLS:
    regress = sm.graphics.plot_regress_exog(multlm,  c )
    regress.show()

influence =sm.graphics.influence_plot(multlm,size=2)    
