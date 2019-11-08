# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:18:43 2019

@author: leomf
"""


import wbdata
import pandas as pd
import pandas_profiling
import datetime
import pandas_datareader
import urllib3


### fecthing country names from IMF database to get all ISO codes
countrycodes = pd.read_csv('coucodes.csv', delimiter = ';', encoding = "ISO-8859-1")
countrycodes = countrycodes[["IMF Name", "ISO Code", "ISO2 Code"]]
countrycodes.reset_index(level=0, inplace=True)
countrycodes = countrycodes.rename(columns={'IMF Name': 'country'})

countries = ["Brazil",  "Mexico", "India",  "Russia", "Switzerland", "Uruguay",
             "Korea", "Thailand", "Ecuador", "Bolivia", "Costa Rica", "Colombia", "Paraguay",
             "Chile", "South Africa", "Taiwan", "Turkey", "Ukraine", "Nigeria", "Indonesia",
             "Bangladesh", "Philippines", "Pakistan", "Egypt", "Ethiopia", "Vietnam", "Tanzania",
             "Myanmar", "Algeria", "Sudan", "Uganda", "Morocco", "Saudi Arabia", "Uzbekistan", "Malaysia",
             "Afghanistan", "Ghana"]

#Argentina, Venezuela e Iraq are not working - Monthly
#
countrycodes = countrycodes[countrycodes["country"].isin(countries)]

#downloading data from worldbank

from pandas_datareader import wb

worldbank_data = pandas_datareader.wb.download(indicator = ["DPANUSLCU", "TOT", "TOTRESV","UNEMPSA_", "IPTOTSAKD", "DXGSRMRCHSAKD","DSTKMKTXD", "DMGSRMRCHSAKD","CPTOTSAXN"], country = countrycodes["ISO Code"], start = 2010, end = 2020, freq = "M")
worldbank_data2 = pandas_datareader.wb.download(indicator = ["DPANUSLCU", "TOT", "TOTRESV","UNEMPSA_",  "IPTOTSAKD", "DXGSRMRCHSAKD","DSTKMKTXD", "DMGSRMRCHSAKD","CPTOTSAXN"], country = countrycodes["ISO Code"], start = 1999, end = 2009, freq = "M")
worldbank_data3 = pandas_datareader.wb.download(indicator = ["DPANUSLCU", "TOT", "TOTRESV","UNEMPSA_","IPTOTSAKD", "DXGSRMRCHSAKD","DSTKMKTXD", "DMGSRMRCHSAKD","CPTOTSAXN"], country = countrycodes["ISO Code"], start = 1990, end = 1998, freq = "M")

worldbank_data = pd.concat([worldbank_data, worldbank_data2])
worldbank_data = pd.concat([worldbank_data, worldbank_data3])

worldbank_data.reset_index(inplace=True)

worldbank_data_conc = pd.merge(worldbank_data, countrycodes, on="country")


# Example: loading IMF data into pandas
#Monthly data
# Import libraries
import requests
import pandas as pd

url = "http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/M."

#Countries
couvalues = list(countrycodes["ISO2 Code"].values)
iso2 = countrycodes["ISO2 Code"].str.cat(sep='+')
iso3 = countrycodes["ISO Code"].str.cat(sep=',')
#Variables
variables = ".RAFAGOLDV_OZT+RAXG_USD+TXG_FOB_XDC+TMG_CIF_XDC+FIMM_PA+PCPI_IX+ENDE_XDC_USD_RATE+AIP_SA_IX.?startPeriod=1990&endPeriod=2030"

url = url+iso2+variables

# Get data from the above URL using the requests package
data = requests.get(url).json()


#### stacking IMF data and creating a dataframe
stack = []
data2 = pd.DataFrame()
for x in range(len(data['CompactData']['DataSet']['Series'])-1):
    data2 = pd.DataFrame(data['CompactData']['DataSet']['Series'][x]["Obs"])[["@OBS_VALUE", "@TIME_PERIOD"]]
    data2["country"] = data['CompactData']['DataSet']['Series'][x]["@REF_AREA"]
    data2["indicator"] = data['CompactData']['DataSet']['Series'][x]["@INDICATOR"]
    stack.append(data2)
stack = pd.concat(stack)


stack = stack.set_index(["@TIME_PERIOD", 'country', 'indicator']).unstack(level=-1)
stack.columns = stack.columns.droplevel(0)

stack.reset_index(inplace=True)


stack = stack.rename(columns={'@TIME_PERIOD': 'year', "country": "ISO2 Code"})
#Next: Analyze the missing data and data range

import missingno as msno
stack_m = pd.merge(stack, countrycodes[["ISO2 Code", "ISO Code"]], on = "ISO2 Code")
msno.matrix(stack_m)

#The same for price commodities index


# Example: loading IMF data into pandas
#Monthly data
# Import libraries
import requests
import pandas as pd

url = "http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/PCTOT/M."

#Countries
couvalues = list(countrycodes["ISO2 Code"].values)
iso2 = countrycodes["ISO2 Code"].str.cat(sep='+')
iso3 = countrycodes["ISO Code"].str.cat(sep=',')
#Variables
variables = ".x_gdp.?startPeriod=1990&endPeriod=2030"

url = url+iso2+variables

# Get data from the above URL using the requests package
data = requests.get(url).json()


#### stacking IMF data and creating a dataframe
stack = []
data2 = pd.DataFrame()
for x in range(len(data['CompactData']['DataSet']['Series'])-1):
    data2 = pd.DataFrame(data['CompactData']['DataSet']['Series'][x]["Obs"])[["@OBS_VALUE", "@TIME_PERIOD"]]
    data2["country"] = data['CompactData']['DataSet']['Series'][x]["@REF_AREA"]
    data2["indicator"] = data['CompactData']['DataSet']['Series'][x]["@INDICATOR"]
    stack.append(data2)
stack = pd.concat(stack)


stack = stack.drop(["indicator"], axis = 1)


stack = stack.rename(columns={'@TIME_PERIOD': 'year', "country": "ISO2 Code", "@OBS_VALUE" : "PCTOT"})
#Next: Analyze the missing data and data range

stack_m2 = stack.drop_duplicates(["year", "ISO2 Code"])

stack_m3 = pd.merge(stack_m, stack_m2, on = ["ISO2 Code", "year"], how = "left")

#Quarterly data


import requests
import pandas as pd

url = "http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/Q."

#Countries
couvalues = list(countrycodes["ISO2 Code"].values)
iso2 = countrycodes["ISO2 Code"].str.cat(sep='+')
iso3 = countrycodes["ISO Code"].str.cat(sep=',')

#Variables
variables = ".NGDP_R_K_IX+PCTOT+BCAXF_BP6_USD+FASMB_XDC.?startPeriod=1990&endPeriod=2020"

url = url+iso2+variables

# Get data from the above URL using the requests package
data = requests.get(url).json()


#### stacking IMF data and creating a dataframe
stack = []
data2 = pd.DataFrame()
for x in range(len(data['CompactData']['DataSet']['Series'])-1):
    data2 = pd.DataFrame(data['CompactData']['DataSet']['Series'][x]["Obs"])[["@OBS_VALUE", "@TIME_PERIOD"]]
    data2["country"] = data['CompactData']['DataSet']['Series'][x]["@REF_AREA"]
    data2["indicator"] = data['CompactData']['DataSet']['Series'][x]["@INDICATOR"]
    stack.append(data2)
stack = pd.concat(stack)


stack = stack.set_index(["@TIME_PERIOD", 'country', 'indicator']).unstack(level=-1)
stack.columns = stack.columns.droplevel(0)

stack.reset_index(inplace=True)


stack = stack.rename(columns={'@TIME_PERIOD': 'year', "country": "ISO2 Code"})
#Next: Analyze the missing data and data range

import missingno as msno
msno.matrix(stack)
stack_q = pd.merge(stack, countrycodes[["ISO2 Code"]], on = "ISO2 Code")


# merging monthly and quarterly

stack_q["year"] = pd.to_datetime(stack_q["year"])
stack_m3["year"] = pd.to_datetime(stack_m3["year"])

stack = pd.merge(stack_m3, stack_q, on = ["year", "ISO2 Code"], how = "left")

### Getting information from OECD

from cif import cif

data_all, subjects_all, measures_all = cif.createDataFrameFromOECD(countries = countrycodes["ISO Code"].tolist(),
                                                                   frequency = 'M', subject = ["SPASTT01"])

oecd = data_all.stack(level = 0)
oecd.columns = oecd.columns.droplevel(0)
oecd = oecd.reset_index()
oecd = oecd.drop(['GP', 'GY'], axis=1)  
oecd["level_0"] = pd.to_datetime(oecd["level_0"])

full = pd.merge(stack, oecd, left_on = ["ISO Code", "year"], right_on = ["country", "level_0"], how = "left")

### Getting BIS data
import certifi
from io import BytesIO
import pandas as pd
from urllib.request import urlopen
from zipfile import ZipFile

z = urlopen('https://www.bis.org/statistics/full_bis_total_credit_csv.zip', cafile = certifi.where())
myzip = ZipFile(BytesIO(z.read())).extract('WEBSTATS_TOTAL_CREDIT_DATAFLOW_csv_col.csv')
credit =pd.read_csv(myzip)


credit = credit[(credit["Borrowing sector"] == "Private non-financial sector") | (credit["Borrowing sector"] == "General government")]
credit = credit[(credit["Lending sector"] == "All sectors")]
credit = credit[(credit["BORROWERS_CTY"].isin(countrycodes["ISO2 Code"]))]
credit = credit[(credit["Unit type"] == "Percentage of GDP")]
credit = credit[(credit["TC_ADJUST"] == "A")]
credit = credit[(credit["Valuation"] == "Market value")]
credit = (credit.set_index(['Borrowing sector', 'BORROWERS_CTY'])
   .rename_axis(['Year'], axis=1)
   .stack()
   .unstack('Borrowing sector')
   .reset_index())
credit = credit.iloc[13:]

z = urlopen('https://www.bis.org/statistics/full_bis_dsr_csv.zip', cafile = certifi.where())
myzip = ZipFile(BytesIO(z.read())).extract('WEBSTATS_DSR_DATAFLOW_csv_col.csv')
debt_service =pd.read_csv(myzip)

debt_service = debt_service[(debt_service["Borrowers"] == "Private non-financial sector")]
debt_service = debt_service[(debt_service["BORROWERS_CTY"].isin(countrycodes["ISO2 Code"]))]
debt_service = (debt_service.set_index(['BORROWERS_CTY', "Borrowers"])
   .rename_axis(['Year'], axis=1)
   .stack()
   .unstack("Borrowers")
   .reset_index())

debt_service = debt_service.iloc[5:]
debt_service.columns = debt_service.columns = ["BORROWERS_CTY", "Year", "debt_service"]

### Getting information from coup risk on political risk

import re
from bs4 import BeautifulSoup, SoupStrainer
import requests

url = "https://oefdatascience.github.io/REIGN.github.io/menu/reign_current.html"

page = requests.get(url)    
data = page.text
soup = BeautifulSoup(data)

links = []
for link in soup.find_all(attrs={'href': re.compile("http")}):
    links.append(link.get('href'))


sub = "https://cdn.rawgit.com/OEFDataScience/REIGN.github.io/gh-pages/data_sets/REIGN"


link = [word for word in links if word.startswith(sub)]

import pandas as pd
couprisk = pd.read_csv(link[0])
couprisk = couprisk[["country", "year", "month", "couprisk"]]
couprisk["month"] = couprisk["month"].astype(int)
couprisk["month"] = couprisk.month.map("{:02}".format)
couprisk["year"] = couprisk["year"].round().astype(int).astype(str) + "-" + couprisk["month"].astype(str)+"-" + "01"

couprisk = couprisk[couprisk["country"].isin(countries)]
couprisk = pd.merge(couprisk, countrycodes, on = "country")
couprisk = couprisk.drop(["index", "ISO Code"], 1)
couprisk = couprisk.drop_duplicates(["year", "country"])

credit["Year"] = pd.to_datetime(credit["Year"], errors = "coerce")
debt_service["Year"] = pd.to_datetime(debt_service["Year"], errors = "coerce")
couprisk["year"] = pd.to_datetime(couprisk["year"], errors = "coerce")


full = pd.merge(full, credit, left_on = ["ISO2 Code", "year"], right_on = ["BORROWERS_CTY", "Year"], how = "left")
full = pd.merge(full, debt_service, left_on = ["ISO2 Code", "year"], right_on = ["BORROWERS_CTY", "Year"], how = "left")
full = pd.merge(full, couprisk, on = ["ISO2 Code", "year"], how = "left")

full

### Creating variables and cleaning the data

#droping duplicated columns
full = full.drop(columns = ["country_x", "Year_x","country_y", "Year_y", "BORROWERS_CTY_x", "BORROWERS_CTY_y",
                            "Year_y","level_0"])

#to numeric
cols = full.columns.drop(["ISO2 Code", "year"])

full[cols] = full[cols].apply(pd.to_numeric, errors='coerce')    

#creating variable

iso = full["ISO2 Code"]

full = full.groupby('ISO2 Code').ffill()

full["ISO2 Code"] = iso

full["gdp_growth"] = full.groupby('ISO2 Code', sort=False).NGDP_R_K_IX.apply(
     lambda x: x.pct_change(12))

full.rename(columns={'Private non-financial sector':'credit_private', 'General government': 'credit_government'}, inplace=True)

full["credit_growth"] = full.groupby('ISO2 Code', sort=False).credit_private.apply(
     lambda x: x.pct_change(12))

full["inflation"] = full.groupby('ISO2 Code', sort=False).PCPI_IX.apply(
     lambda x: x.pct_change(12))

full["stock_growth"]  = full.groupby('ISO2 Code', sort=False).IXOB.apply(
     lambda x: x.pct_change(1))


full["exchange_change"] = full.groupby('ISO2 Code', sort=False).ENDE_XDC_USD_RATE.apply(
     lambda x: x.pct_change(1))

full["exchange_12a"] = full.groupby('ISO2 Code', sort=False).ENDE_XDC_USD_RATE.apply(
     lambda x: x.pct_change(12))

full["industrial_growth"] = full.groupby('ISO2 Code', sort=False).AIP_SA_IX.apply(
     lambda x: x.pct_change(1))

full["commodities_growth"] = full.groupby('ISO2 Code', sort=False).PCTOT.apply(
     lambda x: x.pct_change(12))

full["reserves_gdp"] = full["RAXG_USD"]/full["NGDP_R_K_IX"]

full["imports_gdp"] = full["TMG_CIF_XDC"]/full["NGDP_R_K_IX"]

full["exports_gdp"] = full["TXG_FOB_XDC"]/full["NGDP_R_K_IX"]


#Counting missing values by country
full2 = full
g = full2.groupby('ISO2 Code')
g.count().rsub(g.size(), axis=0)

#Dropping columns used to create the new variables
#Scaling

full3 = full2.drop(["ISO Code", "FASMB_XDC", "credit_government", "RAFAGOLDV_OZT",  "TXG_FOB_XDC", "TMG_CIF_XDC", "RAXG_USD", "PCTOT", "AIP_SA_IX", "ENDE_XDC_USD_RATE", "PCPI_IX", "IXOB"], 1 )
full3 = full3.groupby('ISO2 Code').transform(lambda x: (x - x.mean()) / x.std())
full3["ISO2 Code"] = full2["ISO2 Code"]
full3["year"] = full2["year"]
full3["year2"] = full3["year"].dt.strftime('%Y%m%d').astype(float)
full3 = pd.get_dummies(full3, columns=['ISO2 Code'], prefix = ['Country'])
full3 = full3.set_index("year")

#Scaling
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#scaler.fit(full3)
#full4 = pd.DataFrame(scaler.transform(full3), columns=full3.columns, index = full3.index)

#Imputing missing data with KNN

from fancyimpute import IterativeImputer, IterativeSVD, MatrixFactorization,NuclearNormMinimization, BiScaler, KNN, SoftImpute

X_filled_knn = KNN().fit_transform(full3)

full_fill = pd.DataFrame(data=X_filled_knn, columns=full3.columns, index = full3.index)
full2 = full2.set_index("year")
full_fill["ISO2 Code"] = full2["ISO2 Code"]
full_fill["exchange_change"] = full2["exchange_change"]
full_fill["exchange_12a"] = full2["exchange_12a"]


import statsmodels.api as sm

### Creating FX gap variable

groups = full_fill.groupby('ISO2 Code')

group_keys = list(groups.groups.keys())


bs = pd.DataFrame()

for key in group_keys:

    g = groups.get_group(key).copy()
    target = g['exchange_change']

    cycle, trend = sm.tsa.filters.hpfilter(target, lamb=400000)

    g['fx_gap'] = trend
    bs = bs.append(g)

full_fill = bs

### Creating credit gap variable
groups = full_fill.groupby('ISO2 Code')

group_keys = list(groups.groups.keys())


bs2 = pd.DataFrame()

for key in group_keys:

    g = groups.get_group(key).copy()
    target = g['credit_private']

    cycle, trend = sm.tsa.filters.hpfilter(target, lamb=400000)

    g['credit_gap'] = trend
    bs2 = bs2.append(g)

full_fill = bs2

import numpy as np

full_fill["dummy"] = np.where(full_fill['exchange_change'] >0.1, 1, 0) #10 percent variation

#full_fill["threshold"] = full_fill.groupby("ISO2 Code")["exchange_change"].transform("mean") + 2*(full_fill.groupby("ISO2 Code")["exchange_change"].transform("std"))
#full_fill["dummy"] = np.where(full_fill['exchange_change'] > full_fill["threshold"], 1, 0) # > 2 stdev threshold


crise = full_fill

crise["lag_1"] =crise.groupby("ISO2 Code")["dummy"].shift(-1)
crise["lag_2"] =crise.groupby("ISO2 Code")["dummy"].shift(-2)
crise["lag_3"] =crise.groupby("ISO2 Code")["dummy"].shift(-3)
crise["lag_4"] =crise.groupby("ISO2 Code")["dummy"].shift(-4)
crise["lag_5"] =crise.groupby("ISO2 Code")["dummy"].shift(-5)
crise["lag_6"] =crise.groupby("ISO2 Code")["dummy"].shift(-6)
crise["lag_7"] =crise.groupby("ISO2 Code")["dummy"].shift(-7)
crise["lag_8"] =crise.groupby("ISO2 Code")["dummy"].shift(-8)
crise["lag_9"] =crise.groupby("ISO2 Code")["dummy"].shift(-9)
crise["lag_10"] =crise.groupby("ISO2 Code")["dummy"].shift(-10)
crise["lag_11"] =crise.groupby("ISO2 Code")["dummy"].shift(-11)
crise["lag_12"] =crise.groupby("ISO2 Code")["dummy"].shift(-12)
#crise["yt-1"] =crise.groupby("ISO2 Code")["dummy"].shift(0)
#crise["yt2-1"] = crise["exchange_change"].shift(-1)
crise.fillna(0, inplace=True)


bs2['indicator'] = bs2["lag_1"] + bs2["lag_2"] + bs2["lag_3"] + bs2["lag_4"] + bs2["lag_5"] + bs2["lag_6"] + bs2["lag_7"] + bs2["lag_8"] + bs2["lag_9"] + bs2["lag_10"] + bs2["lag_11"] +bs2["lag_12"]
bs2['indicator2'] = np.where(bs2['indicator'] > 0, 1, 0)


bs2['indicator'] = bs2["lag_1"] +bs2["lag_2"]  + bs2["lag_3"] #+ bs2["lag_4"] + bs2["lag_5"] +bs2["lag_6"]
#bs2['indicator2'] = np.where(bs2['indicator'] > 0, 1, 0)

full_fill = bs2


### Preparing machine learning pipeline 

from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier 
from xgboost import XGBRegressor
from sklearn.neural_network import MLPClassifier

columns = full_fill.columns
columns = columns.drop(["NGDP_R_K_IX", "credit_private", "year2", "dummy", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7", "lag_8", "lag_9", "lag_10", "lag_11", "lag_12", "indicator", "indicator2", "fx_gap", "exchange_change", "month"])


### k-fold

full_fill2 = full_fill.loc["1990-01-01":"2017-12-31"]
full_fill3 = full_fill2
full_fill2 =  full_fill2[full_fill2.columns.drop(list(full_fill2.filter(regex='Country')))]
columns2 = columns.drop(list(full_fill.filter(regex='Country')))
X =  full_fill2[columns2]
Y = full_fill2['indicator2']
X = pd.concat((X, full_fill3.filter(regex='Country')), axis=1)
X = X.drop(["ISO2 Code"], axis =1)

X

### train test split

X_train, X_test, y_train, y_test =train_test_split(X,Y,
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=full_fill2['indicator2'])


# Spot-Check Algorithms
def GetBasedModel():
    basedModels = []
    basedModels.append(('LR'   , LogisticRegression()))
    basedModels.append(('LDA'  , LinearDiscriminantAnalysis()))
    basedModels.append(('KNN'  , KNeighborsClassifier()))
    basedModels.append(('CART' , DecisionTreeClassifier()))
    basedModels.append(('NB'   , GaussianNB()))
    basedModels.append(('SVM'  , SVC(probability=True)))
    basedModels.append(('AB'   , AdaBoostClassifier()))
    basedModels.append(('GBM'  , GradientBoostingClassifier()))
    basedModels.append(('RF'   , RandomForestClassifier()))
    basedModels.append(('ET'   , ExtraTreesClassifier()))
    basedModels.append(('XGB'   , XGBClassifier()))


    
    return basedModels


from sklearn.metrics import confusion_matrix


def BasedLine2(X_train, y_train,models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'roc_auc'

    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds, random_state= 1990)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    return names, results


from plotly import graph_objs as go
import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

class PlotBoxR(object):
    
    
    def __Trace(self,nameOfFeature,value): 
    
        trace = go.Box(
            y=value,
            name = nameOfFeature,
            marker = dict(
                color = 'rgb(0, 128, 128)',
            )
        )
        return trace

    def PlotResult(self,names,results):
        
        data = []

        for i in range(len(names)):
            data.append(self.__Trace(names[i],results[i]))


        py.iplot(data)


from plotly import graph_objs as go

models = GetBasedModel()
names,results = BasedLine2(X_train, y_train,models)
PlotBoxR().PlotResult(names,results)

### 

model =  ExtraTreesClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
confusion_matrix(y_test, y_pred)


### feature importance

clf = ExtraTreesClassifier(random_state= 1984)

clf.fit(X_train, y_train)

# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(8, 20))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])#boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.figure(figsize=(100,150))
plt.show()

## 

X_test["pred"] = y_prob
X_test2 = X_test[X_test["Country_BR"] > 0]
X_test2['pred'].plot(linewidth=5, figsize=(15,15))

# Time-Test

full_fill2 = full_fill.loc["1990-01-01":"2017-12-31"]
full_fill3 = full_fill2
full_fill2 =  full_fill2[full_fill2.columns.drop(list(full_fill2.filter(regex='Country')))]
columns2 = columns.drop(list(full_fill.filter(regex='Country')))
X =  full_fill2[columns2]
Y = full_fill2['indicator2']
X = pd.concat((X, full_fill3.filter(regex='Country')), axis=1)

X = X.drop(["ISO2 Code"], axis =1)

X_train = X.loc["1990-01-01":"2009-12-31"]
X_test = X.loc["2010-01-01":"2017-12-31"]

Y_train = Y.loc["1990-01-01":"2009-12-31"]
Y_test = Y.loc["2010-01-01":"2017-12-31"]


models = GetBasedModel()
names,results = BasedLine2(X_train, Y_train,models)
PlotBoxR().PlotResult(names,results)

model =  XGBClassifier()
model= MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
#confusion_matrix(Y_test, y_pred)

from matplotlib import cm

X_test["pred"] = y_prob
X_test["dummy"] = Y_test
X_test2 = X_test[X_test["Country_RU"] > 0]
X_test2.plot(y= ["dummy","pred"], linewidth=5, figsize=(15,15))

a = full[full["exchange_change"] > 0.1]
a = a[a["year"]> "2010-01-01"]
a["ISO2 Code"]

X_test2["dummy"]

### Time-test (variables in differences)

full_fill2 = full_fill.loc["1990-01-01":"2019-12-31"]
full_fill3 = full_fill2
full_fill2 =  full_fill2[full_fill2.columns.drop(list(full_fill2.filter(regex='Country')))]
columns2 = columns.drop(list(full_fill.filter(regex='Country')))
X =  full_fill2[columns2].groupby("ISO2 Code").diff()
Y = full_fill2['indicator2']
X = pd.concat((X, full_fill3.filter(regex='Country')), axis=1)

X = X.fillna(X.mean())
X = X.fillna(0)

X_train = X.loc["1990-01-01":"2014-12-31"]
X_test = X.loc["2015-01-01":"2019-12-31"]

Y_train = Y.loc["1990-01-01":"2014-12-31"]
Y_test = Y.loc["2015-01-01":"2019-12-31"]


from sklearn.model_selection import TimeSeriesSplit


#cv = GapWalkForward(n_splits=10, gap_size=6, test_size=48)

cv = TimeSeriesSplit(n_splits=10)




params = {
        'max_depth': range (2, 10),
        'n_estimators': [10,  1000],
        'learning_rate': [0.1, 0.01]
        }

clf = GridSearchCV(XGBClassifier(), params, n_jobs=-1, 
                   cv=cv, scoring= "balanced_accuracy",verbose=2, refit=True)



#model =   XGBClassifier()
#model = MLPClassifier(solver='lbfgs', alpha=0.1,hidden_layer_sizes=(52, 42), random_state=1)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]
confusion_matrix(Y_test, y_pred)

#models = GetBasedModel()
#names,results = BasedLine2(X_train, Y_train,models)
#PlotBoxR().PlotResult(names,results)


y_pred = clf.predict(X)
y_prob = clf.predict_proba(X)[:,1]
#confusion_matrix(Y_test, y_pred)

X["pred"] = y_prob
X["dummy"] = Y


X2 = X[X["Country_UA"] > 0]
cycle, trend = sm.tsa.filters.hpfilter(X2["pred"], lamb=5)
X2["hp"] = trend.diff()
X2.plot(y= ["dummy","pred"], linewidth=5, figsize=(15,15))


X_test2["pred"].rolling(window=12).mean()


a = full[full["ISO2 Code"] == "BR"]
a["ENDE_XDC_USD_RATE"]








