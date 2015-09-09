#new learning algorithm - gradient boosting classifier
#runs multiple decision trees, one on top of another, so it gradually improves the result
#ensemble means running different learning algorithms and combining the results

import numpy as np
import pandas
import re
import operator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from data_wrangling import titanic_train as titanic
from data_wrangling import titanic_test

############ extract the title and use it as a feature

#extract the title
def get_title(name):
	pattern = re.compile('([A-Za-z]+)\.')
	match = re.search(pattern, name)
	if match: return match.group(1) 
	return ''

#apply extract function to the dataset
titles = titanic['Name'].apply(get_title)
#title mapping given - alternative version in test.py
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
	titles[titles == k] = v

# .value_counts -> count the number of ocurrences for each uniq entry
#print pandas.value_counts(titles)
titanic['Title'] = titles

#############  extract the family identification and use it as a feature
family_id_mapping = {}

def get_family_id(row):
	last_name = row['Name'].split(',')[0]
	family_id = "{0}{1}".format(last_name, row['FamilySize']) #concatenate last with family size
	if family_id not in family_id_mapping:
		if len(family_id_mapping) == 0:
			current_id=1
		else:
			current_id = max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1
		family_id_mapping[family_id] = current_id
	return family_id_mapping[family_id]

family_ids = titanic.apply(get_family_id, axis=1)
family_ids[titanic['FamilySize'] < 3] = -1
titanic['FamilyId'] = family_ids

#############  the ensemble part

predictorsGBC = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked','Title', 'FamilyId', 'FamilySize', 'NameLength']
predictorsLR = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilyId', 'FamilySize', 'NameLength']
#removed Family Size
algorithms = [
	[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictorsGBC],
	[LogisticRegression(random_state=1), predictorsLR]
]

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = np.array([])
for train, test in kf:
	train_target = titanic['Survived'].iloc[train]
	full_test_predictions = []
	#Make predictions for each algorithm on each fold
	for alg, predictors in algorithms:
		#Fit the algorithm on the training data
		alg.fit(titanic[predictors].iloc[train, :], train_target)
		#Select and predict on the test fold
		#.astype(float) -> convert dataframe to all floats
		test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
		full_test_predictions.append(test_predictions)
	#Use a simple emsenbling scheme -- just average the predictions to get the final classification
	test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
	predictions = np.append(predictions, test_predictions)

#import pdb;pdb.set_trace()
#adjust predictions to yes or no values
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
#compute the accuracy
accuracy = float(sum(predictions == titanic['Survived'])) / len(predictions)
print accuracy

#############  add all new elements to the test set

#add family id
family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test['FamilySize'] < 3] = -1
titanic_test['FamilyId'] = family_ids

#add title
titles = titanic_test['Name'].apply(get_title)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
	titles[titles == k] = v
titanic_test['Title'] = titles

#############  run on test set

full_predictions = []
for alg, predictors in algorithms:
	alg.fit(titanic[predictors], titanic['Survived'])
	predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
	full_predictions.append(predictions)

#weight gradient boosting higher (3/4) since it generates better predictions
predictions = (full_predictions[0]*3 + full_predictions[1])/4

#transform in 1 and 0 and convert to int
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
predictions = predictions.astype(int)

#mke a submission file
submission = pandas.DataFrame({
	'PassengerId': titanic_test['PassengerId'],
	'Survived': predictions
})
submission.to_csv('submit5-ensemble.csv',index=False)