# Earthquake Damage Grade Prediction

## Problem Statement
Determining the degree of damage that is done to buildings post an earthquake can help identify safe and unsafe buildings, thus avoiding death and injuries resulting from aftershocks.

## Data Variables Description:

| Variable  | Description |
| ------------- | ------------- |
| area_assesed  |Indicates the nature of the damage assessment in terms of the areas of the building that were assessed  |
| building_id  | A unique ID that identifies every individual building  |
| damage_grade  |Damage grade assigned to the building after assessment (Target Variable)  |
| district_id  | District where the building is located |
| has_geotechnical_risk  |Indicates if building has geotechnical risks  |
| has_geotechnical_risk_fault_crack  | Indicates if building has geotechnical risks related to fault cracking  |
| has_geotechnical_risk_flood  |Indicates if building has geotechnical risks related to flood |
| has_geotechnical_risk_land_settlement  | Indicates if building has geotechnical risks related to land settlement |
| has_geotechnical_risk_landslide |Indicates if building has geotechnical risks related to landslide  |
| has_geotechnical_risk_liquefaction |Indicates if building has geotechnical risks related to liquefaction |
| has_geotechnical_risk_other |Indicates if building has any other  geotechnical risks  |
| has_geotechnical_risk_rock_fall  |Indicates if building has geotechnical risks related to rock fall |
| has_repair_started  |Indicates if the repair work had started |
| vdcmun_id  | Municipality where the building is located |

## Datatset Link
https://www.hackerearth.com/challenge/competitive/machine-learning-challenge-6-1/problems/

## Evaluation Metric
F1 Score with ‘weighted’ average.

## Models
### 1-D Convolutional Neural Network
```
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
```



