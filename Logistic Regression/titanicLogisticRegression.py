#LOGISTIC REGRESSION
#Logit Regression is used to predict the categorical dependent variable
#with the help of independent variables. to be used in a fuzzy way
#on the other hand
#Linear Regression is used to predict the Continuous dependent variable
#using a given set of independent variables.

#Logistic Regression is designed explicitly for predicting
#probability of an event, Correlation +ve or -ve

import sklearn.linear_model
#print(examples[0])
##fit() :- sequnce of feature, labels return object type LogitRegression
##coef :- Returns the wt of feature
##predict_proba(feature_vector) :- return probabilitites of labels


def buildModel(examples, toPrint = True):
    featureVecs, labels = [], []
    for e in examples:
        featureVecs.append(e.getFeatures())
        labels.append(e.getLabel())

    LogisticRegression = sklearn.linear_model.LogisticRegression
    model = LogisticRegression().fit(featureVecs, labels)

    if toPrint:
        print('model_classes = ', model.classes_)
        for i in range(len(model.coef_)):
            print('For Label', model.classes_[1])
            for j in range(len(model.classes_[0])):
                print(' ', Passenger.featureNames[j], ' = ', model.coef_[0][j])
    return model

            

def applyModel(model, testSet, label, prob = 0.5):
    testFeatureVecs = [e.getFeatures() for e in testSet]
    probs = model.predict_proba(testFeatureVecs)
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i in range(len(probs)):
        if probs[i][1] > prob:
            if testSet[i].getLabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else:
            if testSet[i].getLabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg



def lr(trainingData, testData, prob = 0.5):
    model = buildModel(trainingData, True)
    results = applyModel(model, testData, 'Survived', prob)
    return results

##random.seed(0)
##numSplits = 10
##print('Averagge of ', numSplits, '80/20 splits of LR')
##truePos, falsePos, trueNeg, falseNeg =\
##         randomSplits(examples, lr, numSplits)
##
##print('Average of LOO testing using LR')
##truePos, falsePos, trueNeg, falseNeg =\
##      leaveOneOut(examples, lr)


trainingSet, testSet = split80_20(examples)
model = buildModel(trainingSet, testSet)
print('Try p = 0.1')
truePos, falsePos, trueNeg, falseNeg =\
                   applyModel(model, testSet, 'Survived', 0.1)
getStats(truePos, falsePos, trueNeg, falseNeg)
print('Try p = 0.9')
truePos, falsePos, trueNeg, falseNeg =\
                   applyModel(model, testSet, 'Survived', 0.9)
getStats(truePos, falsePos, trueNeg, falseNeg)



def buildROC(trainingSet, testSet, title, plot = True):
    model = buildModel(trainingSet, True)
    xVals, yVals = [], []
    p = 0.0
    while p <= 1.0:
        truePos, falsePos, trueNeg, falseNeg =\
                               applyModel(model, testSet,
                               'Survived', p)
        xVals.append(1.0 - specificity(trueNeg, falsePos))
        yVals.append(sensitivity(truePos, falseNeg))
        p += 0.01
    auroc = sklearn.metrics.auc(xVals, yVals)
    if plot:
        plt.plot(xVals, yVals)
        plt.plot([0,1], [0,1])
        title = title + '\nAUROC = ' + str(round(auroc,3))
        plt.title(title)
        plt.xlabel('1 - specificity')
        plt.ylabel('Sensitivity')
    return auroc

random.seed(0)
trainingSet, testSet = split80_20(examples)
buildROC(trainingSet, testSet, 'ROC for Predicting Survival, 1 Split')
plt.show()
