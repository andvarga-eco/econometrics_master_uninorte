import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import sys
from sklearn.base import BaseEstimator


file = "https://raw.githubusercontent.com/andvarga-eco/econometrics_master_uninorte/refs/heads/main/Datos%20y%20c%C3%B3digo/geih_24_baq.csv"
df = pd.read_csv(file)
print(df.shape)

#Construir variables

df=df[(df['P6040']>24)&((df['P6040']<65))]
print(df.shape)
df.describe()
print(df.head)

df['lwage']=np.log(df['INGLABO']/df['P6800'])
df['exp']=df['P6040']-16
df['exp2']=df['exp']^2
df['exp3']=df['exp']^3
df['exp4']=df['exp']^4

conditions=[
    df['P3042']<3,
    (df['P3042']>=3) & (df['P3042']<8),
    (df['P3042']>=8) & (df['P3042']<11),
    (df['P3042']>=11) & (df['P3042']<14),
]
choices=[1,2,3,4]

df['educ']=np.select(conditions,choices)
df['educ']=df['educ'].astype('category')
df['educ'].value_counts()

df['P3271']=df['P3271'].astype('category')
df['P3271'].value_counts()

# Create 'sex' variable: 1 if P3271==2, 0 if P3271==1
df['sex'] = np.where(df['P3271'] == 2, 1, np.where(df['P3271'] == 1, 0, np.nan))


df['sector'] = df['RAMA2D_R4'].astype(str).str[:1].astype(int)
df['sector'].value_counts()
df['occ'] = df['OFICIO_C8'].astype(str).str[:1].astype(int)
df['occ'].value_counts()

# Calculate and print mean lwage for each group of P3271
mean_lwage_by_sex = df.groupby('sex')['lwage'].mean()
print("Mean lwage by sex group:")
print(mean_lwage_by_sex)

nocontrol_fit = smf.ols("lwage ~ sex", data=df).fit()
print(nocontrol_fit.summary())

model_flex=('lwage~sex+exp+C(educ)+C(sector)+C(occ)+(exp+exp2+exp3+exp4)*(C(educ)+C(sector)+C(occ))')
control_fit = smf.ols(model_flex, data=df).fit()
control_est = control_fit.params['sex']
control_se = control_fit.HC3_se['sex']
print(f"The estimated sex coefficient is {control_est:.4f} "
      f"and the corresponding robust standard error is {control_se:.4f}")
control_fit.summary()

# Partialling-out

flex_y = "lwage ~exp+C(educ)+C(sector)+C(occ)+(exp+exp2+exp3+exp4)*(C(educ)+C(sector)+C(occ))"  # model for Y
flex_d = "sex ~ exp+C(educ)+C(sector)+C(occ)+(exp+exp2+exp3+exp4)*(C(educ)+C(sector)+C(occ))"  # model for D

# partialling-out the linear effect of W from Y
t_Y = smf.ols(flex_y, data=df).fit().resid

# partialling-out the linear effect of W from D
t_D = smf.ols(flex_d, data=df).fit().resid

# Align indices
t_Y_aligned, t_D_aligned = t_Y.align(t_D, join='inner')

# regression of Y on D after partialling-out the effect of W
partial_fit = sm.OLS(t_Y_aligned, t_D_aligned).fit()
partial_est = partial_fit.params['x1']
print("Coefficient for D via partialling-out " + str(partial_est))

# standard error
partial_se = partial_fit.HC3_se['x1']

# confidence interval
print("95% CI: " + str(partial_fit.conf_int().values[0]))