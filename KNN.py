from numpy import *
import operator 
def createDataSet():
	group = [[1,1.1],[1,1],[0,0],[0,0,1]]
	labels = ['A','A','B','B']
	return group,labels


def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile( inX,(dataSetSize,1)) - dataSetSize
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis = 1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classcount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classcount[voteIlabel] = classcount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classcount.iteritems(),key = operator.itemgetter(1),reverse = true)
	return sortedClassCount[0][0]
	
group,labels=createDataSet()
classify0([0,0],group,labels,3)
