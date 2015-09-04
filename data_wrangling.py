# script to wrangle the data and run basic algorithms
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import numpy as np

########## wrangling training data

#.fillna -> fill not available values
titanic_train = pandas.read_csv("train.csv")

#.fillna -> fill not available values
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())

#.loc -> label based indexing
titanic_train.loc[titanic_train["Sex"] == 'male', "Sex"] = 0
titanic_train.loc[titanic_train["Sex"] == 'female', "Sex"] = 1

#fixing the remaining
titanic_train["Embarked"] = titanic_train["Embarked"].fillna('S')
titanic_train.loc[titanic_train["Embarked"] == 'S', "Embarked"] = 0
titanic_train.loc[titanic_train["Embarked"] == 'C', "Embarked"] = 1
titanic_train.loc[titanic_train["Embarked"] == 'Q', "Embarked"] = 2
titanic_train["Fare"] = titanic_train["Fare"].fillna(titanic_train["Fare"].median())

#generate new columns
titanic_train['FamilySize'] = titanic_train['SibSp'] + titanic_train['Parch']
#.apply -> apply a function to all selected elements
titanic_train['NameLength'] = titanic_train['Name'].apply(lambda x: len(x))

#.fillna -> fill not available values
titanic_test = pandas.read_csv("test.csv")

#.fillna -> fill not available values
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_train["Age"].median())

#.loc -> label based indexing
titanic_test.loc[titanic_test["Sex"] == 'male', "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == 'female', "Sex"] = 1

#fixing the remaining
titanic_test["Embarked"] = titanic_test["Embarked"].fillna('S')
titanic_test.loc[titanic_test["Embarked"] == 'S', "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == 'C', "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == 'Q', "Embarked"] = 2
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

#generate new columns
titanic_test['FamilySize'] = titanic_test['SibSp'] + titanic_test['Parch']
#.apply -> apply a function to all selected elements
titanic_test['NameLength'] = titanic_test['Name'].apply(lambda x: len(x))

