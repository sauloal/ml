#!/usr/bin/env python

from sklearn import tree

FS = [
	'Bumpy',
	'Smooth'
]

LS = [
	"apple",
	"orange"
]

FK = dict([(k,i) for i,k in enumerate(FS)])
LK = dict([(k,i) for i,k in enumerate(LS)])

print FK
print LK

features = [
	[150, FK['Bumpy' ]],
	[170, FK['Bumpy' ]],
	[140, FK['Smooth']],
	[130, FK['Smooth']]
]

labels = [
	LK["orange"],
	LK["orange"],
	LK["apple" ],
	LK["apple" ]
]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print LS[clf.predict([[160,FK['Bumpy']]])[0]]
