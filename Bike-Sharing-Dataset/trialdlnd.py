import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

data = 'hour.csv'

df = pd.read_csv(data)

#print(df[:50])

df[:10].plot(x ='dteday', y = 'cnt')
#plt.show()

dummy_fields = ['season','weathersit','mnth','hr','weekday']

for x in dummy_fields:
	dummies = pd.get_dummies(df[x], prefix = x)
	df = pd.concat([df,dummies],axis = 1)

#print(df[:5])


fields_to_drop = ['instant','dteday','season','weathersit','weekday','atemp','mnth',
                  'workingday', 'hr']
df = df.drop(fields_to_drop, axis= 1)

#print(df[:5])

quant_features = ['casual','registered','cnt','temp','hum','windspeed']
#store scalings in a dictionary so we can convert back later
scaled_features = {}

for x in quant_features:
	mean,std = df[x].mean(),df[x].std()
	scaled_features[x] =[mean,std]
	df.loc[:,x] = (df[x] -mean)/std 

#print(df[:5])

#splitting the data into training, testing and validation sets
#saving the last 21 days
test_data = df[-21*24:]
print(len(test_data))
print (test_data)
df = df[:-21*24]

#separating the data into features and targets
target_fields = ['cnt','casual','registered']
features,targets = df.drop(target_fields, axis =1), df[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis =1), test_data[target_fields]

#separating the features and targets to validation data

train_features,train_targets=features[:-60*24],targets[:-60*24]
val_features,val_targets=features[-60*24:],targets[-60*24:]

