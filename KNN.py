# -*- coding: utf-8 -*-
"""
@author: Swati_Sandhya
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#reading data from file
data = pd.read_csv("parameters.csv")   #or Leaf.csv

#Classification parameters
X = data[['aspect','ecen','foam','narrow','pa','pd','plb','rec']]

#Data preprocessing
#Added these lines to round off data to 6 places, replace all inf values and Nan values to 0
X = X.round(6)
X = X.replace(np.inf, 0)
X = X.replace(np.nan, 0)

#Classification label
Y = data['target']

#splitting the dataset into train and test model by dividing into 70:30 ratio
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.30)

#defining the model with K value as 5
clf = KNeighborsClassifier(n_neighbors = 6)

#fitting the data
clf.fit(x_train,y_train)

#It should be x_test not y_train
pred = clf.predict(x_test)

#Accuracy Matrix
print(accuracy_score(y_test , pred))
print(confusion_matrix(y_test , pred))

