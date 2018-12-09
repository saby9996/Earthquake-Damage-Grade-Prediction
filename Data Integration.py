#Technical Imports
import math
import re
import os
import datetime
import itertools
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt


path='E:/HRC Code Base/Desktop Codes/hackerearth/'
#Reading Data From the CSV Files
train=pd.read_csv(path+"train.csv",sep=',',encoding="utf-8",error_bad_lines=False,low_memory = False)
print(train.shape)
Building_Ownership=pd.read_csv(path+"Building_Ownership_Use.csv",sep=',',encoding="utf-8",error_bad_lines=False,low_memory = False)
print(Building_Ownership.shape)
Building_Structure=pd.read_csv(path+"Building_Structure.csv",sep=',',encoding="utf-8",error_bad_lines=False,low_memory = False)
print(Building_Structure.shape)

#Merging Train_Test + Building Ownership on Building Id
mega_data=pd.merge(train,Building_Ownership, how='left', on='building_id')
print(mega_data.shape)

#Merging Mega Data + Building Structure on Building Id
mega_data=pd.merge(mega_data,Building_Structure, how='left', on='building_id')
print(mega_data.shape)

#Removing Duplicates After Join
mega_data.drop(['district_id_x','district_id_y','vdcmun_id_x','vdcmun_id_y','ward_id_y'], inplace=True, axis=1)

print(mega_data.shape)
print(mega_data.columns)

mega_data.to_csv(path+"mega_data.csv")