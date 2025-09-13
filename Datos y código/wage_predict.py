
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

file = "https://raw.githubusercontent.com/andvarga-eco/econometrics_master_uninorte/refs/heads/main/Datos%20y%20c%C3%B3digo/geih_24_baq.csv"
df = pd.read_csv(file)
print(df.shape)

#Construir variables

df=df[(df['P6040']>24)&((df['P6040']<65))]
print(df.shape)
df.describe()
print(df.head)

df['lwage']=np.log(df['INGLABO']/df['P6800'])
sns.kdeplot(df['lwage'], fill=True)
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

df['sector'] = df['RAMA2D_R4'].astype(str).str[:1].astype(int)
df['sector'].value_counts()
df['occ'] = df['OFICIO_C8'].astype(str).str[:1].astype(int)
df['occ'].value_counts()

# Sample spliting

train, test = train_test_split(df, test_size=0.20, random_state=123)

model_base='lwage~C(P3271)+exp+C(educ)+C(sector)+C(occ)'
base = smf.ols(model_base, data=train)
results_base = base.fit()
print(results_base.summary())

rsquared_base = results_base.rsquared
rsquared_adj_base = results_base.rsquared_adj
mse_base = np.mean(results_base.resid**2)
mse_adj_base = results_base.mse_resid
print(f'Rsquared={rsquared_base:.4f}')
print(f'Rsquared_adjusted={rsquared_adj_base:.4f}')
print(f'MSE={mse_base:.4f}')
print(f'MSE_adjusted={mse_adj_base:.4f}')