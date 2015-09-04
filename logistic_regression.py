# script to wrangle the data and run basic algorithms
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import numpy as np
from data_wrangling import titanic_train
from data_wrangling import titanic_test

########## running linear regression

#set the features which will be used for linear regression
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

#initialize algorithm class
alg = LinearRegression()

#generate cross validation folds. return row indicies corresponding to train and test data
kf = cross_validation.KFold(titanic_train.shape[0], n_folds=3, random_state=1)

#predict
#.iloc -> position based indexing
predictions = np.array([])
for train, test in kf:
	train_predictors = titanic_train[predictors].iloc[train, :]
	train_target = titanic_train["Survived"].iloc[train]
	alg.fit(train_predictors, train_target)
	test_predictions = alg.predict(titanic_train[predictors].iloc[test, :])
	predictions = np.append(predictions,test_predictions, axis=0)

#evaluate errors
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy = sum(predictions == titanic_train["Survived"]) * 1.0 / len(predictions)
print 'linear regression acc: ', accuracy

########## running logistic regression
alg2 = LogisticRegression(random_state=1)
# .cross_validation.cross_val_score -> fits the algorithm, trains the training data and get the accuracy for each fold
# arg1 -> the algorithm
# arg2 -> the training data (X)
# arg3 -> the training results (y)
# arg4 -> number of folds
scores = cross_validation.cross_val_score(alg2, titanic_train[predictors], titanic_train["Survived"], cv=3)
accuracy2 = scores.mean() #takes the mean from the 3 folds
print 'logistic regression acc:', accuracy2

########## running the algorithm on test data
#alg = LogisticRegression(random_state=1)
alg2.fit(titanic_train[predictors], titanic_train["Survived"])
predictions = alg2.predict(titanic_test[predictors])
# test_accuracy = sum(predictions == titanic_test["Survived"]) / len(predictions)
# print 'accuracy on the test data: ', test_accuracy

#creates a dataframe for submission
submission = pandas.DataFrame({
		'PassengerId': titanic_test['PassengerId'],
		'Survived': predictions
	})

submission.to_csv('submit1.csv', index=False)



