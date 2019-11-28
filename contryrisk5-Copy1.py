#!/usr/bin/env python
# coding: utf-8

# # Countryrisk

# In[1]:


import wbdata
import pandas as pd
import pandas_profiling
import datetime
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
import pandas_datareader
import urllib3


# # 1- Extracting and Cleaning the Data from IMF, WB and BIS

# ## Country selection
# 

# In[2]:


### fecthing country names from IMF database to get all ISO codes
countrycodes = pd.read_csv('coucodes.csv', delimiter = ';', encoding = "ISO-8859-1")
countrycodes = countrycodes[["IMF Name", "ISO Code", "ISO2 Code"]]
countrycodes.reset_index(level=0, inplace=True)
countrycodes = countrycodes.rename(columns={'IMF Name': 'country'})

countries = ["Brazil",  "Mexico", "India",  "Russia", "Switzerland", "Uruguay",
             "Korea", "Thailand",  "Bolivia", "Costa Rica", "Colombia", "Paraguay",
             "Chile", "South Africa", "Taiwan", "Turkey", "Ukraine", "Nigeria", "Indonesia",
             "Bangladesh", "Philippines", "Pakistan", "Egypt", "Ethiopia", "Vietnam", "Tanzania",
             "Myanmar", "Algeria", "Sudan", "Uganda", "Morocco", "Uzbekistan", "Malaysia",
             "Afghanistan", "Ghana", "Latvia", "Hong Kong SAR", "Laos", "Singapore", "Poland",
             "Israel", "Czech Republic", "Romania", "New Zealand", "Hungary", "Kazakhstan", "Kenya", "Angola",
             "Ethiopia", "Dominican Republic", "Sri Lanka", "Guatemala", "Bulgaria", "Tanzania", "Belaurus",
            "Croatia", "Uzbekistan", "Syria", "Lebanon", "Slovenia", "Democratic Republic of the Congo",
            "Azerbaijan", "Côte d'Ivoire"]

#Argentina, Australia,  Venezuela e Iraq are not working - Monthly
#
countrycodes = countrycodes[countrycodes["country"].isin(countries)]


# In[3]:


from datetime import datetime
from itertools import product

today = datetime.today()
datem = datetime(today.year, today.month, 1)
str(today.month-1) + "-" + str(today.year)

year = pd.date_range('1990-01-01', end = str(today.month-1) + "-" + str(today.year), freq='MS').strftime('%Y-%m')


base = pd.DataFrame(data=list(product(year, countrycodes["ISO2 Code"])), columns=['year','ISO2 Code'])
base = base.sort_values(by=['ISO2 Code', "year"]).dropna()


# ## WB
# 

# In[4]:


#downloading data from worldbank

#from pandas_datareader import wb

#worldbank_data = pandas_datareader.wb.download(indicator = ["DPANUSLCU", "TOT", "TOTRESV","UNEMPSA_", "IPTOTSAKD", "DXGSRMRCHSAKD","DSTKMKTXD", "DMGSRMRCHSAKD","CPTOTSAXN"], country = countrycodes["ISO Code"], start = 2010, end = 2020, freq = "M")
#worldbank_data2 = pandas_datareader.wb.download(indicator = ["DPANUSLCU", "TOT", "TOTRESV","UNEMPSA_",  "IPTOTSAKD", "DXGSRMRCHSAKD","DSTKMKTXD", "DMGSRMRCHSAKD","CPTOTSAXN"], country = countrycodes["ISO Code"], start = 1999, end = 2009, freq = "M")
#worldbank_data3 = pandas_datareader.wb.download(indicator = ["DPANUSLCU", "TOT", "TOTRESV","UNEMPSA_","IPTOTSAKD", "DXGSRMRCHSAKD","DSTKMKTXD", "DMGSRMRCHSAKD","CPTOTSAXN"], country = countrycodes["ISO Code"], start = 1990, end = 1998, freq = "M")

#worldbank_data = pd.concat([worldbank_data, worldbank_data2])
#worldbank_data = pd.concat([worldbank_data, worldbank_data3])

#worldbank_data.reset_index(inplace=True)

#worldbank_data_conc = pd.merge(worldbank_data, countrycodes, on="country")


# ## IMF

# In[5]:


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

stack_m = pd.merge(base, stack_m, on = ["ISO2 Code", "year"], how = "left")


# In[6]:


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


# In[7]:


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


# In[8]:


# merging monthly and quarterly

stack_q["year"] = pd.to_datetime(stack_q["year"])
stack_m3["year"] = pd.to_datetime(stack_m3["year"])

stack = pd.merge(stack_m3, stack_q, on = ["year", "ISO2 Code"], how = "left")


# ## OECD

# In[9]:


from cif import cif

data_all, subjects_all, measures_all = cif.createDataFrameFromOECD(countries = countrycodes["ISO Code"].tolist(), frequency = 'M', subject = ["SPASTT01"])


# In[10]:


oecd = data_all.stack(level = 0)
oecd.columns = oecd.columns.droplevel(0)
oecd = oecd.reset_index()
oecd = oecd.drop(['GP', 'GY'], axis=1)  
oecd["level_0"] = pd.to_datetime(oecd["level_0"])


# In[11]:


oecd


# In[12]:


full = pd.merge(stack, oecd, left_on = ["ISO Code", "year"], right_on = ["country", "level_0"], how = "left")


# ## BIS

# In[13]:


from io import BytesIO
import pandas as pd
from urllib.request import urlopen
from zipfile import ZipFile

z = urlopen('https://www.bis.org/statistics/full_bis_total_credit_csv.zip')
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


# In[14]:


z = urlopen('https://www.bis.org/statistics/full_bis_dsr_csv.zip')
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


# In[15]:


from bs4 import BeautifulSoup
import requests
import re

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


# In[16]:


couprisk = couprisk[couprisk["country"].isin(countries)]
couprisk = pd.merge(couprisk, countrycodes, on = "country")
couprisk = couprisk.drop(["index", "ISO Code"], 1)
couprisk = couprisk.drop_duplicates(["year", "country"])


# In[17]:


credit["Year"] = pd.to_datetime(credit["Year"], errors = "coerce")
debt_service["Year"] = pd.to_datetime(debt_service["Year"], errors = "coerce")
couprisk["year"] = pd.to_datetime(couprisk["year"], errors = "coerce")


full = pd.merge(full, credit, left_on = ["ISO2 Code", "year"], right_on = ["BORROWERS_CTY", "Year"], how = "left")
full = pd.merge(full, debt_service, left_on = ["ISO2 Code", "year"], right_on = ["BORROWERS_CTY", "Year"], how = "left")
full = pd.merge(full, couprisk, on = ["ISO2 Code", "year"], how = "left")


# In[18]:


full[full["year"] == "2019-10-01"]


# # 2 - Creating variables and cleaning the Data

# In[19]:


#droping duplicated columns
full = full.drop(columns = ["country_x", "Year_x","country_y", "Year_y", "BORROWERS_CTY_x", "BORROWERS_CTY_y", "Year_y","level_0"])


# In[20]:


#to numeric
cols = full.columns.drop(["ISO2 Code", "year"])

full[cols] = full[cols].apply(pd.to_numeric, errors='coerce')


# In[20]:




from missingpy import MissForest
imputer = MissForest()
full_imp = imputer.fit_transform(full)

full = pd.DataFrame(data=full_imp, columns=full.columns, index = full.index)


# In[83]:


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


# In[84]:


#Counting missing values by country
full2 = full
#g = full2.groupby('ISO2 Code')
#g.count().rsub(g.size(), axis=0).to_csv("missings.csv")

full2["year2"] = full2["year"].dt.strftime('%Y%m%d').astype(float)
full2 = pd.get_dummies(full2, columns=['ISO2 Code'], prefix = ['Country'])
full2 = full2.set_index("year")
from numpy import inf

full2[full2 == inf] = 0 




full_fill = pd.DataFrame(data=X_filled_knn, columns=full3.columns, index = full3.index)
full2 = full2.set_index("year")
full_fill["ISO2 Code"] = full2["ISO2 Code"]
full_fill["exchange_change"] = full2["exchange_change"]
full_fill["exchange_12a"] = full2["exchange_12a"]


# In[91]:


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


# In[92]:


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


# # Definition of dependent variable

# In[93]:


import numpy as np

#FIRST DEFINITION
full_fill["dummy"] = np.where(full_fill['exchange_change'] >0.15, 1, 0) #20 percent variation

#SECOND DEFINITION
#full_fill["threshold"] = full_fill.groupby("ISO2 Code")["exchange_change"].transform("mean") + 3*(full_fill.groupby("ISO2 Code")["exchange_change"].transform("std"))
#full_fill["dummy"] = np.where(full_fill['exchange_change'] > full_fill["threshold"], 1, 0) # > 2 stdev threshold

#THIRD DEFINITION
#full_fill["dummy"] = np.where(full_fill['exchange_12a'].shift(-12) >0.3, 1, 0) #50 percent variation in the next 12 months


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


# In[94]:


bs2['indicator'] =  bs2["lag_1"] + bs2["lag_2"] + bs2["lag_3"] + bs2["lag_4"] + bs2["lag_5"] + bs2["lag_6"] + bs2["lag_7"] + bs2["lag_8"] + bs2["lag_9"] + bs2["lag_10"] + bs2["lag_11"] +bs2["lag_12"]
bs2['indicator2'] = np.where(bs2['indicator'] > 0, 1, 0)

#bs2['indicator2'] = crise["dummy"] #Third Definition

#bs2["lag_1"] +
#bs2['indicator'] = bs2["lag_1"] +bs2["lag_2"]  + bs2["lag_3"] + bs2["lag_4"] + bs2["lag_5"] +bs2["lag_6"]
#bs2['indicator2'] = np.where(bs2['indicator'] > 0, 1, 0)


# In[95]:


full_fill = bs2


# In[96]:


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


# In[97]:


columns = full_fill.columns
columns = columns.drop(["NGDP_R_K_IX", "credit_private", "year2", "dummy", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7", "lag_8", "lag_9", "lag_10", "lag_11", "lag_12", "indicator", "indicator2", "fx_gap", "exchange_change", "month"])


# # K- Fold
# 

# In[ ]:


full_fill2 = full_fill.loc["1990-01-01":"2017-12-31"]
full_fill3 = full_fill2
full_fill2 =  full_fill2[full_fill2.columns.drop(list(full_fill2.filter(regex='Country')))]
columns2 = columns.drop(list(full_fill.filter(regex='Country')))
X =  full_fill2[columns2]
Y = full_fill2['indicator2']
X = pd.concat((X, full_fill3.filter(regex='Country')), axis=1)
X = X.drop(["ISO2 Code"], axis =1)


# In[ ]:


X


# In[269]:


### train test split

X_train, X_test, y_train, y_test =train_test_split(X,Y,
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=full_fill2['indicator2'])


# In[270]:


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


# In[271]:


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


# In[272]:


from plotly import graph_objs as go
import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

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


# In[273]:


from plotly import graph_objs as go


# In[274]:


models = GetBasedModel()
names,results = BasedLine2(X_train, y_train,models)
PlotBoxR().PlotResult(names,results)


# In[ ]:


model =  ExtraTreesClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
confusion_matrix(y_test, y_pred)


# In[ ]:


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


# In[ ]:


X_test["pred"] = y_prob
X_test2 = X_test[X_test["Country_BR"] > 0]
X_test2['pred'].plot(linewidth=5, figsize=(15,15))


# # Time-Test

# In[ ]:


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


# In[ ]:


models = GetBasedModel()
names,results = BasedLine2(X_train, Y_train,models)
PlotBoxR().PlotResult(names,results)


# In[ ]:



model =  XGBClassifier()
model= MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
#confusion_matrix(Y_test, y_pred)


# In[ ]:


from matplotlib import cm

X_test["pred"] = y_prob
X_test["dummy"] = Y_test
X_test2 = X_test[X_test["Country_RU"] > 0]
X_test2.plot(y= ["dummy","pred"], linewidth=5, figsize=(15,15))


# In[ ]:


a = full[full["exchange_change"] > 0.1]
a = a[a["year"]> "2010-01-01"]
a["ISO2 Code"]


# In[ ]:


X_test2["dummy"]


# In[49]:


columns


# # Time-Test (variables in differences)

# In[101]:


full_fill2 = full_fill.loc["1995-01-01":"2019-12-31"]
#full_fill2 = full_fill2.sort_index() #For CV purpose
full_fill3 = full_fill2
full_fill2 =  full_fill2[full_fill2.columns.drop(list(full_fill2.filter(regex='Country')))]
columns2 = columns.drop(list(full_fill.filter(regex='Country')))
columns2 = ["inflation", "reserves_gdp", "commodities_growth", "couprisk", "ISO2 Code"]
X =  full_fill2[columns2].groupby("ISO2 Code").diff()

X_lag = full_fill2[columns2].groupby("ISO2 Code").diff().shift(1)
X_lag2 = full_fill2[columns2].groupby("ISO2 Code").diff().shift(2)
X_lag3 = full_fill2[columns2].groupby("ISO2 Code").diff().shift(3)
X_lag4 = full_fill2[columns2].groupby("ISO2 Code").diff().shift(4)
X_lag5 = full_fill2[columns2].groupby("ISO2 Code").diff().shift(5)

X_level = full_fill2[columns2].drop(["ISO2 Code"],1)

X_lag = X_lag.add_suffix('_lag')
X_lag2 = X_lag2.add_suffix('_lag2')
X_lag3 = X_lag3.add_suffix('_lag3')
X_lag4 = X_lag4.add_suffix('_lag4')
X_lag5 = X_lag5.add_suffix('_lag5')

X_level = X_level.add_suffix('_level')




Y = full_fill2['indicator2']
X = pd.concat((X, full_fill3.filter(regex='Country')), axis=1)
X = pd.concat([X, X_lag], axis=1)
#X = pd.concat([X, X_lag2], axis=1)
#X = pd.concat([X, X_lag3], axis=1)
#X = pd.concat([X, X_lag4], axis=1)
#X = pd.concat([X, X_lag5], axis=1)
X = pd.concat([X, X_level], axis=1)

X = X.fillna(X.mean())

#X = X.groupby('ISO2 Code_level').transform(lambda x: (x - x.mean()) / x.std())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns, index = X.index)

X_train = X.loc["1990-01-01":"2009-12-31"]
X_test = X.loc["2010-01-01":"2019-12-31"]

Y_train = Y.loc["1990-01-01":"2009-12-31"]
Y_test = Y.loc["2010-01-01":"2019-12-31"]


# In[106]:


from sklearn.model_selection import TimeSeriesSplit


#cv = GapWalkForward(n_splits=10, gap_size=6, test_size=48)

#sort by before
cv = TimeSeriesSplit(n_splits=4)




#params = {
        #'max_depth': [2,3,5,7],
        #'n_estimators': [50, 100, 200, 500, 1000],
        #'learning_rate': [0.01, 0.1, 0.05, 0.001],
        #'colsample_bytree : [1, 0.8, 0.5]'
        #}

#clf = GridSearchCV(XGBClassifier(), params, n_jobs = -1, 
                   #cv=cv, scoring= "balanced_accuracy",verbose=2, refit=True)



clf =   XGBClassifier(n_estimators = 500,  eval_metric="auc")
#clf = LogisticRegression()
#clf = KNeighborsClassifier()
#clf = MLPClassifier()
#clf = ExtraTreesClassifier()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]
confusion_matrix(Y_test, y_pred)


# In[107]:



full_fill2 = full_fill.loc["1990-01-01":"2019-12-31"]
full_fill3 = full_fill2
full_fill2 =  full_fill2[full_fill2.columns.drop(list(full_fill2.filter(regex='Country')))]
columns2 = columns.drop(list(full_fill.filter(regex='Country')))
columns2 = ["inflation", "reserves_gdp", "commodities_growth", "couprisk", "ISO2 Code"]
X =  full_fill2[columns2].groupby("ISO2 Code").diff()

X_lag = full_fill2[columns2].groupby("ISO2 Code").diff().shift(1)
X_lag2 = full_fill2[columns2].groupby("ISO2 Code").diff().shift(2)
X_lag3 = full_fill2[columns2].groupby("ISO2 Code").diff().shift(3)
X_lag4 = full_fill2[columns2].groupby("ISO2 Code").diff().shift(4)
X_lag5 = full_fill2[columns2].groupby("ISO2 Code").diff().shift(5)

X_level = full_fill2[columns2].drop(["ISO2 Code"],1)

X_lag = X_lag.add_suffix('_lag')
X_lag2 = X_lag2.add_suffix('_lag2')
X_lag3 = X_lag3.add_suffix('_lag3')
X_lag4 = X_lag4.add_suffix('_lag4')
X_lag5 = X_lag5.add_suffix('_lag5')

X_level = X_level.add_suffix('_level')




Y = full_fill2['indicator2']
X = pd.concat((X, full_fill3.filter(regex='Country')), axis=1)
X = pd.concat([X, X_lag], axis=1)
#X = pd.concat([X, X_lag2], axis=1)
#X = pd.concat([X, X_lag3], axis=1)
#X = pd.concat([X, X_lag4], axis=1)
#X = pd.concat([X, X_lag5], axis=1)
X = pd.concat([X, X_level], axis=1)

X = X.fillna(X.mean())

#X = X.groupby('ISO2 Code_level').transform(lambda x: (x - x.mean()) / x.std())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns, index = X.index)


X_train = X.loc["1990-01-01":"2009-12-31"]
X_test = X.loc["2010-01-01":"2019-12-31"]

Y_train = Y.loc["1990-01-01":"2009-12-31"]
Y_test = Y.loc["2010-01-01":"2019-12-31"]

X3 = X 
y_pred = clf.predict(X3)
y_prob = clf.predict_proba(X3)[:,1]
#confusion_matrix(Y_test, y_pred)

X3["pred"] = y_prob
X3["precrisis"] = Y

currency = full2
currency = currency.set_index(["ISO2 Code"], append = True)
currency = currency["ENDE_XDC_USD_RATE"]
X3["iso"] = full_fill["ISO2 Code"]
X3.set_index(["iso"], inplace = True, append = True)
X3 = pd.concat([X3, currency], axis = 1)

crisis = full_fill
crisis = crisis.set_index(["ISO2 Code"], append = True)
crisis = crisis["dummy"]
X3 = pd.concat([X3, crisis], axis = 1)


X2 = X3[X3["Country_BR"] > 0]
X2["ENDE_XDC_USD_RATE"] = (X2["ENDE_XDC_USD_RATE"])/np.nanmax(X2["ENDE_XDC_USD_RATE"])
cycle, trend = sm.tsa.filters.hpfilter(X2["pred"], lamb=5)
X2["hp"] = trend
#X2.plot(y= ["dummy","pred", "ENDE_XDC_USD_RATE"], linewidth=5, figsize=(15,15))




import plotly.express as px
import plotly.graph_objects as go

X2.reset_index(inplace = True)

fig = go.Figure()



fig.add_trace(go.Bar(x=X2["year"],
                y=X2["dummy"],
                name='Crisis',
                marker_color='rgb(264, 45, 45)',
                width = 2678400000
                    ))

fig.add_trace(go.Bar(x=X2["year"],
                y=X2["precrisis"],
                name='Precrisis',
                marker_color='rgb(120, 120, 120)',
                marker_line_color = 'rgb(128, 128, 128)',
                opacity = 0.4,
                width = 2678400000
                    ))
                
fig.add_trace(go.Scatter(
                x=X2["year"],
                y=X2['pred'],
                name="Prob",
                line_color='deepskyblue',
                opacity=1,
                line=dict(color='deepskyblue', width=4
                              )))

fig.add_trace(go.Scatter(
                x=X2["year"],
                y=X2["ENDE_XDC_USD_RATE"],
                name="Exchange Rate",
                line_color='dimgray',
                opacity=0.6,
                line=dict(color='firebrick', width=4)))


fig.update_layout(
    autosize=False,
    width=1000,
    height=600)


# In[65]:


full_fill.to_csv("full.csv")


# In[114]:


X3["iso"]


# In[810]:


last = X3.reset_index()
last = last[last["year"] == "2019-10-01"]
last.sort_values(by=['pred'])


# In[165]:


import matplotlib.pyplot as plt

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


# In[ ]:


a = full[full["ISO2 Code"] == "BR"]
a["ENDE_XDC_USD_RATE"]


# full_fill

# In[383]:


full_fill2 = full_fill.loc["1990-01-01":"2019-12-31"]
full_fill2 = full_fill2.drop("exchange_12a", axis = 1)

full_fill2 = full_fill2[full_fill2["Country_BR"] > 0]
full_fill2 =  full_fill2[full_fill2.columns.drop(list(full_fill2.filter(regex='Country')))]
columns2 = columns.drop(list(full_fill3.filter(regex='Country')))

X =  full_fill2[columns2]
#X = X.groupby(["ISO2 Code"])
X = X.drop("ISO2 Code",1)
Y = full_fill2['indicator2']


X = X.fillna(X.mean())


X_train = X.loc["1990-01-01":"2009-12-31"]
X_test = X.loc["2010-01-01":"2019-12-31"]

Y_train = Y.loc["1990-01-01":"2009-12-31"]
Y_test = Y.loc["2010-01-01":"2019-12-31"]

model =   XGBClassifier()


model= MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
y_prob = model.predict_proba(X)[:,1]

confusion_matrix(Y_test, y_pred)


# In[384]:


X


# In[385]:


import plotly.express as px
import plotly.graph_objects as go

X_test = full_fill2


X_test["pred"] = y_prob
X_test["dummy2"] = Y_test
X_test["pred_ma"] = X_test.pred.rolling(window=6).mean()

cycle, trend = sm.tsa.filters.hpfilter(X_test["pred"], lamb=5)
X_test["hp"] = trend
#X_test.plot(y= ["dummy","pred"], linewidth=5, figsize=(15,15))
X_test = X_test.reset_index()

crisis_line = X_test[X_test["dummy"] == 1]

fig = go.Figure()
fig.add_trace(go.Scatter(
                x=X_test.year,
                y=X_test['pred'],
                name="Prob",
                line_color='deepskyblue',
                opacity=1,
                line=dict(color='deepskyblue', width=4
                              )))

fig.add_trace(go.Scatter(
                x=X_test.year,
                y=X_test['dummy2'],
                name="12 months before currency crisis",
                line_color='dimgray',
                opacity=0.8,
                line=dict(color='firebrick', width=4,
                              dash='dot')))

fig.add_trace(go.Scatter(
                x=X_test.year,
                y=X_test['dummy'],
                name="crisis",
                line_color='red',
                opacity=0.8,
                line=dict(color='firebrick', width=4,
                              dash='dash')))


# In[ ]:


y_prob


# In[ ]:


aa =full_fill[full_fill["indicator2"] == 1]
aa = aa.loc["2009-01-01":"2019-12-31"]
aa["ISO2 Code"].unique()


# In[ ]:


X.plot(y= ["gdp_growth"], linewidth=5, figsize=(15,15))


# In[ ]:


full3


# 
# ## time validation
# 

# In[ ]:


#full_fill["dummy"] = np.where(full_fill['exchange_change'] > 0.05, 1, full_fill["dummy"])
full_fill["dummy"] = np.where(full_fill['exchange_change'] > 0.1, 1, 0)
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

crise["indicator"] = crise[["lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7", "lag_8", "lag_9", "lag_10", "lag_11", "lag_12"]].max(axis=1)
#crise["indicator"] = crise[["lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6"]].max(axis=1)
#crise["indicator"] = crise[["lag_1", "lag_2", "lag_3"]].max(axis=1)

full_fill = crise


# In[ ]:


columns = full_fill.columns
columns = columns.drop(["ENDE_XDC_USD_RATE", "NGDP_R_K_IX", "IXOB", "PCPI_IX", "credit_private", "year2", "dummy", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7", "lag_8", "lag_9", "lag_10", "lag_11", "lag_12", "indicator", "fx_gap", "exchange_change", "exchange_12a"])


# In[ ]:


full_fill2 = full_fill.loc["1995-01-01":"2009-12-31"]
full_fill2 = full_fill2.drop('exchange_12a', axis = 1)
full_fill2 = full_fill[full_fill["Country_BR"] > 0]
full_fill2 =  full_fill2[full_fill2.columns.drop(list(full_fill2.filter(regex='Country')))]
columns2 = columns.drop(list(full_fill.filter(regex='Country')))
full_fill2 = full_fill2.loc["1995-01-01":"2009-12-31"]

X =  full_fill2[columns2]

X = X.groupby(["ISO2 Code"])
#X = X.drop("ISO2 Code",1)
Y = full_fill2['indicator']



X = X.fillna(X.mean())


# In[ ]:


from sklearn.model_selection import TimeSeriesSplit


#cv = GapWalkForward(n_splits=10, gap_size=6, test_size=48)

cv = TimeSeriesSplit(n_splits=10)



parameters = {'solver': ['lbfgs'], 'max_iter': [1000,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(5, 50)}
clf = GridSearchCV(MLPClassifier(random_state = 1984), parameters, n_jobs=-1, cv = cv)

model = clf.fit(X, Y)


# In[ ]:


y_prob = model.predict_proba(X)[:,1]
X2 = X
X2["pred"] = y_prob
X2["dummy"] = Y

X2.plot(y= ["dummy","pred"], linewidth=5, figsize=(15,15))


# In[ ]:


full_fill2 = full_fill.loc["1990-01-01":"2019-12-31"]
full_fill2 = full_fill2.drop('exchange_12a', axis = 1)
full_fill2 = full_fill[full_fill["Country_BR"] > 0]
full_fill2 =  full_fill2[full_fill2.columns.drop(list(full_fill2.filter(regex='Country')))]
columns2 = columns.drop(list(full_fill.filter(regex='Country')))

X =  full_fill2[columns2]
#X = X.drop('exchange_12a', axis = 1)
X = X.groupby(["ISO2 Code"]).diff()
#X = X.drop("ISO2 Code",1)
Y = full_fill2['indicator']



X = X.fillna(X.mean())


# In[ ]:


y_prob = model.predict_proba(X)[:,1]
X2 = X
X2["pred"] = y_prob
cycle, trend = sm.tsa.filters.hpfilter(X2["pred"], lamb=5)
X2["hp"] = trend
X2["dummy"] = Y

X2.plot(y= ["dummy","pred"], linewidth=5, figsize=(15,15))


# In[ ]:



full_fill.to_csv("output.csv") 


# In[ ]:




