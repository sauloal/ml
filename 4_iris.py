#!/usr/bin/env python

DEBUG = False

clf_type = "tree"
clf_type = "kneighbour"

import numpy as np
from sklearn.datasets   import load_iris
if   clf_type == "tree":
	from sklearn           import tree                 as my_classifier
	clf = my_classifier.DecisionTreeClassifier()
elif clf_type == "kneighbour":
	from sklearn.neighbors import KNeighborsClassifier as my_classifier
	clf = my_classifier()



iris     = load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

clf.fit( X_train, y_train )

pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print accuracy_score(y_test, pred)

quit()

test_idx = [0,50,100]

#training data
train_target = np.delete(iris.target, test_idx        )
train_data   = np.delete(iris.data  , test_idx, axis=0)

#testing data
test_target  = iris.target[test_idx]
test_data    = iris.data[  test_idx]

print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[0], iris.target_names[iris.target[0]]

if DEBUG:
	for i in xrange(len(iris.target)):
		if i == 0:
			print ("{:20s} "*len(iris.feature_names)).format(*iris.feature_names), "{:20}".format("TARGET")
		print ("{:20} "*len(iris.feature_names)).format(*iris.data[i]), "{:>20}".format(iris.target_names[iris.target[i]])


clf = tree.DecisionTreeClassifier()
clf.fit( train_data, train_target )

print "Expected :", test_target, "[", ", ".join([str(iris.target_names[i]) for i in test_target]), "]"
pred = clf.predict( test_data )
print "Predicted:", pred       , "[", ", ".join([str(iris.target_names[i]) for i in pred       ]), "]"


#viz code
from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf,
	out_file      = dot_data,
	feature_names = iris.feature_names,
	class_names   = iris.target_names,
	filled        = True,
	rounded       = True,
	impurity      = False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
print graph
graph[0].write_pdf("iris.pdf")
