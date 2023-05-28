# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Verilerin Import Edilmesi
dataset = pd.read_csv("breast_cancer.csv")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

#Train ve Test Setleri
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0)

#Logistic Regression Modelinin Eğitilmesi
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Model Tahmin Denemesi
y_pred = classifier.predict(X_test)
y_pred_prob = classifier.predict_proba(X_test)


#Consufion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
AccuracyScore = accuracy_score(y_test, y_pred)

#k-Fold Cross Validation Yöntemiyle Modelin Doğruluğunun Test Edilmesi
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv=10)
AccuraciesMean = accuracies.mean()*100
StandardDeviation = accuracies.std()*100













