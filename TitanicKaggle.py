# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:40:04 2019

@author: Casey Pham

Titanic Kaggle Challenge

Machine learning exercise to intro into Kaggle, and to practice basic data 
science and machine learning techniques.
"""

# %% Import Modules

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# %% Import Data

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

print(train_data.head())
print(test_data.head())

# %% Basic Exploration

cols = train_data.columns.values
print(cols)

sex = train_data.groupby("Sex").count()
print("\nCount of People\n",sex["Embarked"])

survived = train_data["Survived"].sum()
print("\nNumber of survivors:", survived)

survived_sex = train_data.groupby("Sex").sum()
print("\nNumber of survivors by sex:\n", survived_sex["Survived"])

# %% Data Prep

# Map strings for female and male to 0 and 1
mapping = {"female" : 0, "male" : 1}
train_data = train_data.replace(mapping)

# Separate survived column into its own data
train_y = train_data["Survived"]
train_x = train_data.drop("Survived", axis = 1)

x = pd.DataFrame(train_x[["Sex", "Pclass"]])

# Separate features to make it a little easier later to 
Pclass = pd.DataFrame(train_data["Pclass"])
Sex = pd.DataFrame(train_data["Sex"])
Age = pd.DataFrame(train_data["Age"]).fillna(0)
Fare = pd.DataFrame(train_data["Fare"])
Sibsp = pd.DataFrame(train_data["SibSp"])
Parch = pd.DataFrame(train_data["SibSp"])
Ticket = pd.DataFrame(train_data["Ticket"])
Fare = pd.DataFrame(train_data["Fare"])
Cabin = pd.DataFrame(train_data["Cabin"])           # Unusable, lots of NaN
Embarked = pd.DataFrame(train_data["Embarked"])

# %% Train Test Split & Multinomial Naive Bayes

def multinb(x, y):
    """
    
    This function performs the required functions for fitting and prediction a 
    Multinomial Naive Bayes
    from given x and y datasets.
    
    Args:
        x (array-like): independent data
        y (array-like): target data
        
    Return:
        score (float): Mean accuracy of the model on the given test and target 
        data
    
    """
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,
                                                        random_state = 0)
    
    # Fit and predict model
    multinb = MultinomialNB()
    multinb.fit(X_train, y_train)
    
    predicted = multinb.predict(X_test)
    predicted
    
    multinb.predict(X_test)
    score = multinb.score(X_test, y_test)
    
    # Plot
    # x_axis = range(len(X_test))
    #
    # fig,ax = plt.subplots(figsize=(15,10))
    # ax.scatter(x_axis, predicted, alpha = 0.3)
    # ax.scatter(x_axis, y_test, alpha = 0.3)
    
    return score

# %%
    
Sex_score = multinb(Sex, train_y)
print("Sex:", Sex_score)
    
Age_score = multinb(Age, train_y)
print("Age:", Age_score)

AgeSex_score = multinb(pd.concat([Age, Sex], axis = 1), train_y)
print("Age + Sex:", AgeSex_score)

Pclass_score = multinb(Pclass, train_y)
print("Pclass:", Pclass_score)

AgeSexPclass_score = multinb(pd.concat([Age, Sex, Pclass], axis = 1), train_y)
print("Age + Sex + Pclass:", AgeSexPclass_score)









