import pandas
import numpy as np
from data_wrangling import titanic_train
from data_wrangling import titanic_test
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import re

############ basic random forest setup

predictors = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp']
#predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

#random forest algorithm initialization
#n_estimators = number of trees
#min_samples_split = min number of rows to make the split
#min_samples leaf = min number of samples at end of tree branch
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

#tweak parameters:
# enhance accuracy by increasing the number of trees
# reduce overfitting by increasing min samples on the split and on the leaf
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

############ more data wrangling

predictors = predictors + ['FamilySize', 'NameLength']

"""
#extract the title
def get_title(name):
	pattern = re.compile('([A-Za-z]+)\.')
	match = re.search(pattern, name)
	if match: return match.group(1) 
	return ''

#apply extract function to the dataset
titles = titanic_train['Name'].apply(get_title)
#title mapping given - alternative version in test.py
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
	titles[titles == k] = v

# .value_counts -> count the number of ocurrences for each uniq entry
#print pandas.value_counts(titles)

titanic_train['Titles'] = titles
"""

############ run the cross validations with K-fold
X = titanic_train[predictors]
y = titanic_train["Survived"]
scores = cross_validation.cross_val_score(alg, X, y, cv=3)
accuracy = scores.mean()
print accuracy

########### run prediction and save
alg.fit(titanic_train[predictors], titanic_train["Survived"])
predictions = alg.predict(titanic_test[predictors])

#creates a dataframe for submission
submission = pandas.DataFrame({
		'PassengerId': titanic_test['PassengerId'],
		'Survived': predictions
	})

submission.to_csv('submit2.csv', index=False)




