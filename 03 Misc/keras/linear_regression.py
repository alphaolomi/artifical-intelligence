# impoting

import numpy as np
import matploit.pyplot as plt
import pandas as pd
import seaborn as sns

%matploit inline


# 2
# Load the dataset and extract independent and dependent variables
# Importing the dataset
companies  = pd.read_csv('c:/Users/alpha/Desktop/1000_Companies.csv')
x= companies.iloc[:,:-1].values
y= companies.iloc[:,:4].values

companies.head()

# 3
# Data Visualization
# Building the correlain matrxi

sns.heatmap(companies.corr())



#
# Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OndeHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])

onehotencoder = OndeHotEncoder( cetegorical_features = [3] )
X = onehotencoder.fit_transform(X).toarray()


# check the outpt
print(X)
X = X[:,1:]


# Split the dataset into Training set and Test set
from sklearn.model_skeleton imprt train_test_split
X_train, Y_test, y_train, x_train = train_test_split(X,Y, test_size=0.2, random_state =0 )



# Fiiting Mupltiple Linear Regresion to the Training data
from sklearn.linear_model import LineaRegression
regressor = LineaRegression()
regressor.fit(X_train, Y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)
# Output : jupyeter inline print
# print(Y_pred)
y_pred


# 9.
# Calculating the Coeffieents and Interceptps
## Calculating the coeffieents
print(regressor.coef_)

## Calculating the interceptps
print(regressor.intercepts_ )


# 10
# Evalutating the model

## Calculating the R squared values
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# NOTE: R squaredvalue of 0.91 proves the model good
