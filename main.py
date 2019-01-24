import numpy as np
import pandas as pd
import scipy
import sklearn
import os
import re
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
from sklearn.model_selection import train_test_split
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

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import MaxPooling1D
from keras.layers import Activation
from keras import optimizers
from keras.layers import Conv1D
from keras.optimizers import SGD #Stochastic Gradient Descent Optimizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding

from keras.applications import inception_v3

import livelossplot
from livelossplot import PlotLossesKeras
plot_losses = livelossplot.PlotLossesKeras()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#dataset injestion
path_to_dir="C:/Users/nEW u/MACHINE LEARNING/project/Earthquake-Damage-Grade-Prediction/Data/"
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

#-----------------------------------------------------------------------------

train=pd.read_csv(path_to_dir+'train.csv',sep=',',encoding="utf-8",error_bad_lines=False,low_memory = False)
building_ownership=pd.read_csv(path_to_dir+'Building_Ownership_Use.csv',sep=',',encoding="utf-8",error_bad_lines=False,low_memory = False)
building_structure=pd.read_csv(path_to_dir+'Building_Structure.csv',sep=',',encoding="utf-8",error_bad_lines=False,low_memory = False)

#Merging Building Details
building_det=pd.merge(building_ownership,building_structure,on=['building_id','vdcmun_id','district_id','ward_id'],how='inner')

#Merging Building Details With Train Set
mega_data=pd.merge(train,building_det,on=['building_id','vdcmun_id','district_id'],how='inner')

mega_data.head()

#PreProcessing
#--------------------------
#To check the Number of Families Having Null
mega_data.columns[mega_data.isna().any()]

#Imputation of Columns with Mean and Mode
from sklearn.preprocessing import Imputer
imput = Imputer(strategy='most_frequent')
imput1 = Imputer(strategy='mean')
mega_data['count_families'] = imput1.fit_transform(mega_data[['count_families']]).astype('int')
mega_data['has_repair_started'] = imput.fit_transform(mega_data[['has_repair_started']]).astype('int')

"""From the Features it can be inferred that the Grade of Damage can be predicted only one the basis of the risk factor of a particular building and also based on the specifications of the building.
Therefore Going With Following Variables
A Risk Variable for Determining the Summation of Risk Factors
An Average, Count and Ratio Feature Based on Municipal, District and Ward For Coorelating [A different Method of String Concatenation Could be used, But Not in the Favour of That]
The Height and And No of Floors of a Building"""

mega_data['Risk_Factor']=mega_data['has_geotechnical_risk_other']+mega_data['has_geotechnical_risk_liquefaction']+ mega_data['has_geotechnical_risk_landslide'] + mega_data['has_geotechnical_risk_flood'] + mega_data['has_geotechnical_risk_rock_fall'] + mega_data['has_geotechnical_risk_land_settlement'] + mega_data['has_geotechnical_risk_fault_crack']
mega_data['Risk_Factor'].describe()

#District Level
# Sum Feature and Average Feature
district_list=mega_data.district_id.unique().tolist()

mega_data['Risk_Factor_District_Sum']=0
mega_data['Risk_Factor_District_Average']=0

for dist in district_list:
    temp_new = pd.DataFrame()
    temp_new = mega_data.loc[mega_data.district_id == dist, :]
    temp_new = temp_new.reset_index()
    temp_new.drop('index', axis=1, inplace=True)
        
    sum_feat=temp_new['Risk_Factor'].sum()
    avg_feat=sum_feat/len(temp_new)
   
    mega_data['Risk_Factor_District_Sum']=np.where(mega_data['district_id']==dist, sum_feat, mega_data['Risk_Factor_District_Sum'])
    mega_data['Risk_Factor_District_Average']=np.where(mega_data['district_id']==dist, avg_feat, mega_data['Risk_Factor_District_Average'])

#Municipal Level
municipal_list=mega_data.vdcmun_id.unique().tolist()

mega_data['Risk_Factor_Municipal_Sum']=0
mega_data['Risk_Factor_Municipal_Average']=0

for mun in municipal_list:
    temp_new = pd.DataFrame()
    temp_new = mega_data.loc[mega_data.vdcmun_id == mun, :]
    temp_new = temp_new.reset_index()
    temp_new.drop('index', axis=1, inplace=True)
        
    sum_feat=temp_new['Risk_Factor'].sum()
    avg_feat=sum_feat/len(temp_new)
   
    mega_data['Risk_Factor_Municipal_Sum']=np.where(mega_data['vdcmun_id']==mun, sum_feat, mega_data['Risk_Factor_Municipal_Sum'])
    mega_data['Risk_Factor_Municipal_Average']=np.where(mega_data['vdcmun_id']==mun, avg_feat, mega_data['Risk_Factor_Municipal_Average'])    

#Ward Level
ward_list=mega_data.ward_id.unique().tolist()

mega_data['Risk_Factor_Ward_Sum']=0
mega_data['Risk_Factor_Ward_Average']=0

for ward in ward_list:
    temp_new = pd.DataFrame()
    temp_new = mega_data.loc[mega_data.ward_id == ward, :]
    temp_new = temp_new.reset_index()
    temp_new.drop('index', axis=1, inplace=True)
        
    sum_feat=temp_new['Risk_Factor'].sum()
    avg_feat=sum_feat/len(temp_new)
   
    mega_data['Risk_Factor_Ward_Sum']=np.where(mega_data['ward_id']==mun, sum_feat, mega_data['Risk_Factor_Ward_Sum'])
    mega_data['Risk_Factor_Ward_Average']=np.where(mega_data['ward_id']==mun, avg_feat, mega_data['Risk_Factor_Ward_Average'])

#Floor Difference
mega_data['Difference_In_Floors'] = mega_data['count_floors_pre_eq']-mega_data['count_floors_post_eq']

#Height Difference
mega_data['Difference_In_Height'] =mega_data['height_ft_pre_eq']-mega_data['height_ft_post_eq']
#Normal Label Encoding
grade_enc = {'Grade 1':1,'Grade 2':2,'Grade 3':3,'Grade 4':4,'Grade 5':5}
labels_norm=mega_data['damage_grade'].map(grade_enc)

# On-hot Encoding
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
labels_one_hot= encoder.fit_transform(labels_norm)
### Removing Unwanted Columns
mega_data = mega_data.drop(columns=['building_id','damage_grade'])
## Transforming All the Columns having Object Type to One Hot Encoding
for col in mega_data.columns:
    if mega_data[col].dtype == 'object':
        dummy=pd.get_dummies(mega_data[col],prefix=col)
        mega_data=pd.concat([mega_data, dummy], axis=1)
        mega_data.drop(col, axis=1,inplace=True)
        
#Removing ID Columns
mega_data.drop(['district_id','vdcmun_id','ward_id'], axis=1, inplace=True)
        
#Non-Parametric Models
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(mega_data, labels_norm, test_size = 0.3, random_state =9, stratify=labels_norm)      
        
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
scaler.fit(train_features)  
train_features = scaler.transform(train_features)  
test_features = scaler.transform(test_features)        

model=RandomForestClassifier(random_state=22, n_estimators=200, max_depth=10)
model.fit(train_features,train_labels)
predictions=model.predict(test_features)
print(accuracy_score(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly
plotly.tools.set_credentials_file(username='c.sabyasachi99', api_key='y5FSl1jIheriCgKbK3Ff')

features=mega_data.columns
importances = model.feature_importances_
std = np.std([model.feature_importances_ for tree in model.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(train_features.shape[1]):
    print( (features[indices[f]], importances[indices[f]]))

trace = go.Bar(x=features, y=importances[indices],
               marker=dict(color='red'),
               error_y=dict(visible=True, arrayminus=std[indices]),
               opacity=0.5)

layout = go.Layout(title="Feature importances")
fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
        
import xgboost
from xgboost import XGBClassifier
modelxg=XGBClassifier(max_depth=10, learning_rate=0.01, n_estimators=200, objective='multi:softmax', booster='gbtree')
modelxg.fit(train_features,train_labels)
predictions=modelxg.predict(test_features)
print(accuracy_score(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))

#Convolutional Neural Network 
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(mega_data, labels_one_hot, test_size = 0.3, random_state =9, stratify=labels_one_hot)
test_labels

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
scaler.fit(train_features)  
train_features = scaler.transform(train_features)  
test_features = scaler.transform(test_features)  

train_features=train_features.reshape(train_features.shape[0], train_features.shape[1], 1)
test_features=test_features.reshape(test_features.shape[0], test_features.shape[1], 1)

print(train_features.shape)
print(test_features.shape)

print(train_labels.shape)
print(test_labels.shape)

import livelossplot
from livelossplot import PlotLossesKeras
plot_losses = livelossplot.PlotLossesKeras()

def createModel():
    model = Sequential()
    model.add(Conv1D(128, 10, input_shape=(98,1,), activation='relu'))
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))  
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    
    #stochastic gradient descent
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

model = createModel()
history=model.fit(train_features,train_labels, epochs=20, batch_size=10,validation_data=(test_features,test_labels),callbacks=[plot_losses])

scores = model.evaluate(test_features, test_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Artificial Neural Network

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(mega_data, labels_one_hot, test_size = 0.3, random_state =9, stratify=labels_one_hot)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
scaler.fit(train_features)  
train_features = scaler.transform(train_features)  
test_features = scaler.transform(test_features) 

print(train_features.shape)
print(test_features.shape)
print(train_labels.shape)
print(test_labels.shape)

import livelossplot
from livelossplot import PlotLossesKeras
plot_losses = livelossplot.PlotLossesKeras()

def create_network():
    model = Sequential()
    model.add(Dense(40, input_shape=(98,), activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='softmax'))
        
    #stochastic gradient descent
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

model3 = create_network()
model3.fit(train_features,train_labels, epochs=100, batch_size=100,validation_data=(test_features,test_labels), callbacks=[plot_losses])
scores = model3.evaluate(test_features, test_labels)
print("\n%s: %.2f%%" % (model3.metrics_names[1], scores[1]*100))

        
        




