from io import BytesIO
import base64
import matplotlib 
import pandas as pd
import pickle
import numpy as np
import scipy as sp
import math
import re
import pymysql as db
import os
#import cv2
import sklearn
import datetime
#from matplotlib import pyplot as plt
#from sshtunnel import SSHTunnelForwarder
from sklearn import model_selection, svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import classification_report, average_precision_score, mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
#from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve, roc_auc_score

#from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
import time

from pyglmnet import GLM
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
#from mlxtend.classifier import StackingClassifier

from scipy.io import loadmat
import scipy


#For Seaborn
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 30,10
import matplotlib.pyplot as plt



#For Arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

#For NLP
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer

#For Bayesian Optimization
#import bayes_opt
#from bayes_opt import BayesianOptimization


#Dask Framework
import dask.dataframe as dd

#For View
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

import warnings
warnings.filterwarnings('ignore')

path='C:/Users/nEW u/MACHINE LEARNING/project/Earthquake-Damage-Grade-Prediction/Data/'
mega_data=pd.read_csv(path+"mega_data.csv",sep=',',encoding="utf-8",error_bad_lines=False,low_memory = False)
print(mega_data.shape)

#### Details of the files are as follows: 

#### train.csv : 

|Varaible| Descrition|
|--------|-----------|
|area_assesed|Indicates the nature of the damage assessment in terms of the areas of the building that were assessed|
|building_id|A unique ID that identifies every individual building|
|damage_grade|Damage grade assigned to the building after assessment (Target Variable)|
|district_id|District where the building is located|
|has_geotechnical_risk|Indicates if building has geotechnical risks|
|has_geotechnical_risk_fault_crack|Indicates if building has geotechnical risks related to fault cracking|
|has_geotechnical_risk_flood|Indicates if building has geotechnical risks related to flood|
|has_geotechnical_risk_land_settlement|Indicates if building has geotechnical risks related to land settlement|
|has_geotechnical_risk_landslide|Indicates if building has geotechnical risks related to landslide|
|has_geotechnical_risk_liquefaction|Indicates if building has geotechnical risks related to liquefaction|
|has_geotechnical_risk_other|Indicates if building has any other  geotechnical risks|
|has_geotechnical_risk_rock_fall|Indicates if building has geotechnical risks related to rock fall|
|has_repair_started|Indicates if the repair work had started|
|vdcmun_id|Municipality where the building is located|

#### Building_Ownership_Use.csv: 

|Varaible|Description|
|--------|-----------|
|building_id|A unique ID that identifies every individual building|
|district_id|District where the building is located|
|vdcmun_id|Municipality where the building is located|
|ward_id|Ward Number in which the building is located|
|legal_ownership_status|Legal ownership status of the land in which the building was built|
|count_families|Number of families in the building|
|has_secondary_use|indicates if the building is used for any secondary purpose|
|has_secondary_use_agriculture|indicates if the building is secondarily used for agricultural purpose|
|has_secondary_use_hotel|indicates if the building is secondarily used as hotel|
|has_secondary_use_rental|indicates if the building is secondarily used for rental purpose|
|has_secondary_use_institution|indicates if the building is secondarily used for institutional purpose|
|has_secondary_use_school|indicates if the building is secondarily used as school|
|has_secondary_use_industry|indicates if the building is secondarily used for industrial purpose|
|has_secondary_use_health_post|indicates if the building is secondarily used as health post|
|has_secondary_use_gov_office|indicates if the building is secondarily used as government office|
|has_secondary_use_use_police|indicates if the building is secondarily used as police station|
|has_secondary_use_other|indicates if the building is secondarily used for other purpose|


#### Building_Structure.csv

|Variable|Description|
|--------|-----------|
|building_id|A unique ID that identifies every individual building|
|district_id|District where the building is located|
|vdcmun_id|Municipality where the building is located|
|ward_id|Ward Number in which the building is located|
|count_floors_pre_eq|Number of floors that the building had before the earthquake|
|count_floors_post_eq|Number of floors that the building had after the earthquake|
|age_building|Age of the building (in years)|
|plinth_area_sq_ft|Plinth area of the building (in square feet)|
|height_ft_pre_eq|Height of the building before the earthquake (in feet)|
|height_ft_post_eq|Height of the building after the earthquake (in feet)|
|land_surface_condition|Surface condition of the land in which the building is built	|
|foundation_type|Type of foundation used in the building|
|roof_type|Type of roof used in the building|
|ground_floor_type|Ground floor type|
|other_floor_type|Type of construction used in other floors (except ground floor and roof)|
|position|Position of the building|
|plan_configuration|Building plan configuration|
|has_superstructure_adobe_mud|indicates if the superstructure of the building is made of Adobe/Mud|
|has_superstructure_mud_mortar_stone|indicates if the superstructure of the building is made of Mud Mortar - Stone|
|has_superstructure_stone_flag| indicates if the superstructure of the building is made of Stone|
|has_superstructure_mud_mortar_brick|indicates if the superstructure of the building is made of Cement Mortar - Stone|
|has_superstructure_cement_mortar_brick|indicates if the superstructure of the building is made of Mud Mortar - Brick|
|has_superstructure_timber|indicates if the superstructure of the building is made of Timber|
|has_superstructure_bamboo|indicates if the superstructure of the building is made of Bamboo|
|has_superstructure_rc_non_engineered|indicates if the superstructure of the building is made of RC (Non Engineered)|
|has_superstructure_rc_engineered|indicates if the superstructure of the building is made of RC (Engineered)|
|has_superstructure_other| indicates if the superstructure of the building is made of any other material|
|condition_post_eq|Actual contition of the building after the earthquake|

## Features on the Basis of District Level

##  has_ features 

has_features=['has_geotechnical_risk_fault_crack','has_geotechnical_risk_flood','has_geotechnical_risk_land_settlement',
       'has_geotechnical_risk_landslide', 'has_geotechnical_risk_liquefaction','has_geotechnical_risk_other', 
        'has_geotechnical_risk_rock_fall','has_repair_started','has_secondary_use', 'has_secondary_use_agriculture',
       'has_secondary_use_hotel', 'has_secondary_use_rental','has_secondary_use_institution', 'has_secondary_use_school',
       'has_secondary_use_industry', 'has_secondary_use_health_post','has_secondary_use_gov_office',
        'has_secondary_use_use_police','has_secondary_use_other','has_superstructure_adobe_mud',
       'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag','has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick','has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered','has_superstructure_rc_engineered',
        'has_superstructure_other']
district_list=mega_data.district_id.unique().tolist()

### COUNT
#Creation has_features_count Columns
for i in has_features:
    mega_data[i+'_district_count']=0
for i in has_features:
    for dist in district_list:
        temp_new = pd.DataFrame()
        temp_new = mega_data.loc[mega_data.district_id == dist, :]
        temp_new = temp_new.reset_index()
        temp_new.drop('index', axis=1, inplace=True)
        
        has_obj=temp_new[temp_new[i]==1]
        value=len(has_obj)
        
        mega_data[i+'_district_count']=np.where(mega_data['district_id']==dist, value, mega_data[i+'_district_count'])

### RATIO 
#### [No of Records Having 1] / [Total Number of Records]
#Creation has_features_ratio Columns
for i in has_features:
    mega_data[i+'_district_ratio']=0        
for i in has_features:
    for dist in district_list:
        temp_new = pd.DataFrame()
        temp_new = mega_data.loc[mega_data.district_id == dist, :]
        temp_new = temp_new.reset_index()
        temp_new.drop('index', axis=1, inplace=True)
        
        has_obj=temp_new[temp_new[i]==1]
        value=len(has_obj)/len(temp_new)
        
        mega_data[i+'_district_ratio']=np.where(mega_data['district_id']==dist, value, mega_data[i+'_district_ratio'])

### AVERAGE RATIO
#### { [Number of Records Having 0] / [Total Number of Records] } / { [Number of Records Having 1] / [Total Number of Records] }
#Creation has_features_ratio Columns
for i in has_features:
    mega_data[i+'_district_average_ratio']=0
for i in has_features:
    for dist in district_list:
        temp_new = pd.DataFrame()
        temp_new = mega_data.loc[mega_data.district_id == dist, :]
        temp_new = temp_new.reset_index()
        temp_new.drop('index', axis=1, inplace=True)
        
        has_obj=temp_new[temp_new[i]==1]
        nos_obj=temp_new[temp_new[i]==0]
        value=(len(has_obj)/len(temp_new))/(len(nos_obj)/len(temp_new))
        
        mega_data[i+'_district_average_ratio']=np.where(mega_data['district_id']==dist, value, mega_data[i+'_district_average_ratio'])
## Features on the Basis of Municipal Level

has_features=['has_geotechnical_risk_fault_crack','has_geotechnical_risk_flood','has_geotechnical_risk_land_settlement',
       'has_geotechnical_risk_landslide', 'has_geotechnical_risk_liquefaction','has_geotechnical_risk_other', 
        'has_geotechnical_risk_rock_fall','has_repair_started','has_secondary_use', 'has_secondary_use_agriculture',
       'has_secondary_use_hotel', 'has_secondary_use_rental','has_secondary_use_institution', 'has_secondary_use_school',
       'has_secondary_use_industry', 'has_secondary_use_health_post','has_secondary_use_gov_office',
        'has_secondary_use_use_police','has_secondary_use_other','has_superstructure_adobe_mud',
       'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag','has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick','has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered','has_superstructure_rc_engineered',
        'has_superstructure_other']
municipal_list=mega_data.vdcmun_id.unique().tolist()

### COUNT
#Creation has_features_count Columns
for i in has_features:
    mega_data[i+'_municipal_count']=0
for i in has_features:
    for mun in municipal_list:
        temp_new = pd.DataFrame()
        temp_new = mega_data.loc[mega_data.vdcmun_id == mun, :]
        temp_new = temp_new.reset_index()
        temp_new.drop('index', axis=1, inplace=True)
        
        has_obj=temp_new[temp_new[i]==1]
        value=len(has_obj)
        
        mega_data[i+'_municipal_count']=np.where(mega_data['vdcmun_id']==mun, value, mega_data[i+'_municipal_count'])

### RATIO 
#### [No of Records Having 1] / [Total Number of Records]
#Creation has_features_ratio Columns
for i in has_features:
    mega_data[i+'_municipal_ratio']=0
for i in has_features:
    for mun in municipal_list:
        temp_new = pd.DataFrame()
        temp_new = mega_data.loc[mega_data.vdcmun_id == mun, :]
        temp_new = temp_new.reset_index()
        temp_new.drop('index', axis=1, inplace=True)
        
        has_obj=temp_new[temp_new[i]==1]
        value=len(has_obj)/len(temp_new)
        
        mega_data[i+'_municipal_ratio']=np.where(mega_data['vdcmun_id']==mun, value, mega_data[i+'_municipal_ratio'])

### AVERAGE RATIO
#### { [Number of Records Having 0] / [Total Number of Records] } / { [Number of Records Having 1] / [Total Number of Records] }
#Creation has_features_ratio Columns
for i in has_features:
    mega_data[i+'_municipal_average_ratio']=0
for i in has_features:
    for mun in municipal_list:
        temp_new = pd.DataFrame()
        temp_new = mega_data.loc[mega_data.vdcmun_id == mun, :]
        temp_new = temp_new.reset_index()
        temp_new.drop('index', axis=1, inplace=True)
        
        has_obj=temp_new[temp_new[i]==1]
        nos_obj=temp_new[temp_new[i]==0]
        if(len(nos_obj)==0):
            value=0
        else:
            value=(len(has_obj)/len(temp_new))/(len(nos_obj)/len(temp_new))
        
        mega_data[i+'_municipal_average_ratio']=np.where(mega_data['vdcmun_id']==mun, value, mega_data[i+'_municipal_average_ratio'])
### AVERAGE NUMBER OF FLOORS BEFORE EARTHQUAKE  DISTRICT, MUNICIPAL and WARD
#List On District, Municipal and Ward Level
district_list=mega_data.district_id.unique().tolist()
municipal_list=mega_data.vdcmun_id.unique().tolist()
ward_list=mega_data.ward_id_x.unique().tolist()
#District LEvel
mega_data['avg_floors_before_quake_district']=0
for dist in district_list:
        temp_new = pd.DataFrame()
        temp_new = mega_data.loc[mega_data.district_id == dist, :]
        temp_new = temp_new.reset_index()
        temp_new.drop('index', axis=1, inplace=True)
        
        number_of_floors=temp_new['count_floors_pre_eq'].sum()
        value=number_of_floors/len(temp_new)
        
        mega_data['avg_floors_before_quake_district']=np.where(mega_data['district_id']==dist, value, mega_data['avg_floors_before_quake_district'])
#Municipal LEvel
mega_data['avg_floors_before_quake_municipal']=0
for mun in municipal_list:
        temp_new = pd.DataFrame()
        temp_new = mega_data.loc[mega_data.vdcmun_id == mun, :]
        temp_new = temp_new.reset_index()
        temp_new.drop('index', axis=1, inplace=True)
        
        number_of_floors=temp_new['count_floors_pre_eq'].sum()
        value=number_of_floors/len(temp_new)
        
        mega_data['avg_floors_before_quake_municipal']=np.where(mega_data['vdcmun_id']==mun, value, mega_data['avg_floors_before_quake_municipal'])
#ward level
mega_data['avg_floors_before_quake_ward']=0
for ward in ward_list:
        temp_new = pd.DataFrame()
        temp_new = mega_data.loc[mega_data.ward_id_x == ward, :]
        temp_new = temp_new.reset_index()
        temp_new.drop('index' ,  axis=1 , inplace = True)
        
        number_of_floors=temp_new['count_floors_pre_eq'].sum()
        value = number_of_floors/len(temp_new)
        mega_data['avg_floors_before_quake_municipal']=np.where(mega_data['ward_id_x']==ward, value, mega_data['avg_floors_before_quake_municipal'])


### AVERAGE NUMBER OF FLOORS AFTER EARTHQUAKE  DISTRICT, MUNICIPAL and WARD
#District LEvel
mega_data['avg_floors_after_quake_district']=0
for dist in district_list:
        temp_new = pd.DataFrame()
        temp_new = mega_data.loc[mega_data.district_id == dist, :]
        temp_new = temp_new.reset_index()
        temp_new.drop('index', axis=1, inplace=True)
        
        number_of_floors=temp_new['count_floors_post_eq'].sum()
        value=number_of_floors/len(temp_new)
        
        mega_data['avg_floors_after_quake_district']=np.where(mega_data['district_id']==dist, value, mega_data['avg_floors_after_quake_district'])
#Municipal LEvel
mega_data['avg_floors_after_quake_municipal']=0
for mun in municipal_list:
        temp_new = pd.DataFrame()
        temp_new = mega_data.loc[mega_data.vdcmun_id == mun, :]
        temp_new = temp_new.reset_index()
        temp_new.drop('index', axis=1, inplace=True)
        
        number_of_floors=temp_new['count_floors_post_eq'].sum()
        value=number_of_floors/len(temp_new)
        
        mega_data['avg_floors_after_quake_municipal']=np.where(mega_data['vdcmun_id']==mun, value, mega_data['avg_floors_after_quake_municipal'])


### AVERAGE NUMBER OF FAMILIES IN A DISTRICT & MUNICIPAL
mega_data['avg_number_families_in_municipal']=0
mega_data['number_of_families_in_municipal']=0
for mun in municipal_list:
    temp_new=pd.DataFrame()
    temp_new=mega_data.loc[mega_data.vdcmun_id== mun,:]
    temp_new=temp_new.reset_index()
    temp_new.drop('index', axis=1, inplace=True)
    
    number_of_families=temp_new['count_families'].sum()
    value=number_of_families/len(temp_new)
    
    mega_data['avg_number_families_in_municipal']=np.where(mega_data['vdcmun_id']==mun, value, mega_data['avg_number_families_in_municipal'])
    mega_data['number_of_families_in_municipal']=np.where(mega_data['district_id']==dist, number_of_families, mega_data['number_of_families_in_municipal'])    
    



