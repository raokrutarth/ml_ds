import pandas as pd
import csv
import numpy as np
import string
import random

import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

def writeResult(res, outFile):
	w = csv.writer(open(outFile, "wb"))
	w.writerow(['id', 'sentiment'])
	for key, val in res.items():
		w.writerow([key, val])
	print( '[+] Saved results to {}'.format(outFile))
	return

def makeDF(fileName, test=False):
	rows = []
	f = open(fileName, 'r') #, encoding="utf8")
	reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
	for row in reader:
		if row[0] == 'id':
			headers = row
		else:
			if test and len(row) != 2: # some corrupted entries in the test dataset
				print row
				row[1] = row[1] + row[2]; del row[2]
			rows.append(np.array(row))
	dm = np.array(rows)
	return pd.DataFrame(dm, columns=headers)

def makeModel(trainDF):
	pipeline = Pipeline([
	('vect', TfidfVectorizer(stop_words='english')),
	('clf', LogisticRegression())
	])
	parameters = {
		'vect__max_df': (0.25, 0.5, 0.75, 1.00),
		'vect__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5) ),
		'vect__use_idf': (True, False),
		'clf__C': (0.1, 1, 5, 10, 15, 20, 25, 30, 35),
	}
	X, y = trainDF['text'], trainDF['sentiment'].as_matrix()
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

	grid_search = GridSearchCV(pipeline, parameters, n_jobs=5, verbose=1, scoring='accuracy')
	grid_search.fit(X, y)
	print ('best score: %0.3f' % grid_search.best_score_)
	print ('best parameters set:')
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print ('\t %s: %r' % (param_name, best_parameters[param_name])   )

	predictions = grid_search.predict(X)
	print ('Accuracy:', accuracy_score(y, predictions))
	return grid_search


def main():
	trainDF = makeDF('Headline_Trainingdata.csv')
	testDF = makeDF('Headline_Testingdata.csv', test=True)
	model = makeModel(trainDF)
	preds = model.predict(testDF['text'])

	print('[+] Predictions on test dataset made')
	res = {}
	i =0
	for id in testDF['id']:
		res[id] = preds[i]
		i += 1
	if i != len(testDF):
		print("[-] mismatch number of predictions")
	writeResult(res, 'testResult.csv')

if __name__ == '__main__':
	main()

warnings.filterwarnings("ignore",category=DeprecationWarning)