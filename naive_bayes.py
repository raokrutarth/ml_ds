import argparse
import pandas as pd
import sys
import random
import operator
import numpy as np
from collections import Counter, defaultdict
from math import log10
from sklearn.naive_bayes import GaussianNB

# ------------------------ Naive Bayes -------------------------

def naive_bayes(trainM, classAssignments):
    gaunb = GaussianNB()
    model = gaunb.fit(trainM, classAssignments)
    return model

def predict(model, newVec):
    prediction = model.predict([newVec])
    return prediction

# ------------------------ Setup -------------------------
def readData(fileName):
    data = pd.read_csv(fileName, sep=',', quotechar='"', header=0, engine='python')
    dm =  data.as_matrix()
    vecMatrix = dm[:, :-1] # remove the classification
    classification = dm[:, -1] # store classifications
    return dm, vecMatrix, classification

def getBinary(dataMatrix):
    df = pd.DataFrame(dataMatrix)
    dBin = pd.get_dummies(df, columns=df.columns.values)
    # remove goodForGroups_0 (14_0) as a vector value
    dBin = dBin.drop(columns=[ dBin.columns[-2] ])
    # print(dBin)
    dM = dBin.as_matrix()
    return dM

def nbc(trainM, testM):
    trainBinM = getBinary(trainM)
    testBinM = getBinary(testM)
    # print(trainBinM[:, :-1])
    model = naive_bayes(trainBinM[:, :-1], trainBinM[:, -1])
    z1loss = zeroOneLoss(testBinM, model)
    print('ZERO-ONE LOSS=%.4f'%z1loss)

# ------------------------ Tests -------------------------
def zeroOneLoss(matrix, model):
    nCorrect = 0.0
    for i in range(len(matrix)):
        pred = predict(model, matrix[i][:-1])
        # print(pred)
        if pred[0]==matrix[i][-1]:
            nCorrect+=1.0
    return 1.00 - (nCorrect/float(len(matrix)))

def squaredLoss(matrix, weights):
    return 0.456587


def majority(matrix):
    classAssignments = matrix[:, -1]
    classCount={}
    for vote in classAssignments:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
        key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def baseline(dataM, label):
    nCorrect = 0.0
    for i in range(len(dataM)):
        if label==dataM[i][-1]:
            nCorrect+=1.0
    return 1.00 - (nCorrect/float(len(dataM)))

# ------------------------ Analysis -------------------------
def crossValidate(trainM, outFile):
    print('Cross-validation check')
    ps = [0.01, 0.10, 0.50, 0.70]
    numChecks = 10

    n = len(trainM)
    w = [0.8]* (len(trainM[0]) -1)
    mf = majority(trainM)

    percs = []; ts = []; z1losses = []; slosses = []; base = []
    for perc in ps:
        frac = int(n*perc)
        avgZ1Loss = 0.0
        avgSqLoss = 0.0
        avgBase = 0.0
        for i in range(numChecks):
            trainPart = random.sample(trainM, frac)
            testPart = random.sample(trainM, n-frac)

            cp, probs = naive_bayes(trainPart[:, :-1], trainPart[:, -1])
            z1loss = zeroOneLoss(testPart, cp, probs)
            sloss = squaredLoss(testPart, cp, probs)
            bl = baseline(testPart, mf)
            print('%.2f percent train data -> %d training entries and %d test entries:'
            ' Zero One Loss=%.6f, Squared Loss: %0.6f, baseline: %.6f'% ((100*perc),frac,n-frac,z1loss,sloss,bl) )
            avgZ1Loss = avgZ1Loss + z1loss
            avgSqLoss = avgSqLoss + sloss
            avgBase = avgBase + bl

        avgZ1Loss = avgZ1Loss/float(numChecks)
        avgBase = avgBase/float(numChecks)
        avgSqLoss = avgSqLoss/float(numChecks)
        percs.append((100*perc))
        ts.append(frac)

        z1losses.append(avgZ1Loss)
        slosses.append(avgSqLoss); base.append(bl)
        avgZ1Loss = 0.0; avgBase = 0.0; avgSqLoss = 0.0
    # save analysis to csv file
    df = pd.DataFrame.from_items([('PercentageTraining', percs), ('TrainingSetSize', ts),
    ('MeanZeroOneLoss', z1losses), ('MeanSquaredLoss', slosses), ('BaselineLoss', base)])
    df.to_csv(outFile,index=False)


# ------------------------ Main -------------------------
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('trainingDataFile', help='Location of training data')
    parser.add_argument('testingDataFile', help='Location of test data')
    args = parser.parse_args()
    return args

def main():
    args = getArgs()
    trainM, trainVec, trainClass = readData(args.trainingDataFile)
    testM, testVec, testClass = readData(args.testingDataFile)
    nbc(trainM, testM)
    # crossValidate(trainM, 'avgCross.csv')

if __name__ == "__main__":
    main()