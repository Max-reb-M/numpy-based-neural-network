import numpy as np
from matplotlib import pyplot as plt
from initialization import activationFunc, countActivations, generateWeightsMatrix, meanCostFunction, buildGradient, getMeanGradient
import copy

dataSet = [{'input': np.array([0, 1, 1]),'output': np.array([1])},
  {'input': np.array([0, 0, 0]), 'output': np.array([0])},
  {'input': np.array([1, 0, 0]), 'output': np.array([0])},
  {'input': np.array([1, 1, 0]), 'output': np.array([0])},
  {'input': np.array([0, 1, 0]), 'output': np.array([0])},
  {'input': np.array([1, 0, 1]), 'output': np.array([1])},
  {'input': np.array([1, 1, 1]), 'output': np.array([1])},
  {'input': np.array([0, 0, 1]), 'output': np.array([0])}]
inputsArr = [0, 1]
expectedOutput = [1]
matrixMap = [3, 3, 3, 1]
weightsMatrix = generateWeightsMatrix(matrixMap)
print('Hipothesis:  ', countActivations([1,1,1], matrixMap, weightsMatrix))
def modelLearning(dataSet, weights, matrixMap, learningRate = 0.01, convergenceRate=0.001, breakLineIteration=1000):
    activationsMatrix = countActivations(inputsArr, matrixMap, weights)
    costDelta = 1
    iterationNum = 0
    valuesOfCostByIteration = np.array([])
    #print('Weights:  ', weightsMatrix)
    #while (costDelta > convergenceRate or costDelta < -convergenceRate) and len(valuesOfCostByIteration) < breakLineIteration:
    while (costDelta > convergenceRate or costDelta < -convergenceRate) and iterationNum < breakLineIteration:
        costDelta = 0 
        costBeforeUpgrade = meanCostFunction(dataSet, matrixMap, weights)
        allGradients = []
        for dataItem in dataSet:
            allGradients.append(buildGradient(matrixMap, weights, dataItem['input'], dataItem['output']))
            #print('32 allGradients:   ', buildGradient(matrixMap, weights, dataItem['input'], dataItem['output'], dataSet, 100000))
        weightsGradient = getMeanGradient(allGradients)
        #print(weightsGradient)
        for layerInd in range(len(weights)):
            for rightNeuronConnectionInd in range(len(weights[layerInd])):
                for leftNeuronConnectionInd in range(len(weights[layerInd][rightNeuronConnectionInd])):
                    weights[layerInd][rightNeuronConnectionInd][leftNeuronConnectionInd] -= weightsGradient[layerInd][rightNeuronConnectionInd][leftNeuronConnectionInd] * learningRate
        #weights -= weightsGradient * learningRate
        iterationNum += 1
        #activationsMatrix = countActivations(dataItem['input'], matrixMap, weights)
        costAfterUpgrade = meanCostFunction(dataSet, matrixMap, weights)
        valuesOfCostByIteration = np.append(valuesOfCostByIteration, costAfterUpgrade)
        costDelta = costAfterUpgrade - costBeforeUpgrade
        print('Cost of one example in the func:   ', costAfterUpgrade)
    return {'weights': weights, 'costsArrPerIteration': valuesOfCostByIteration}


ml = modelLearning(dataSet, weightsMatrix, matrixMap, 0.5, 0.0001, 200)
weightsMatrix = ml['weights']
costGraph = ml['costsArrPerIteration']

activationsMatrix = countActivations(np.array([0, 0, 0]), matrixMap, weightsMatrix)
print('Cost at the end  ', meanCostFunction(dataSet, matrixMap, weightsMatrix))
print('Activations:  ', countActivations(np.array([0, 0, 0]), matrixMap, weightsMatrix))
print('Weights:  ', weightsMatrix)
print('Hipothesis:  ', countActivations([1,1,1], matrixMap, weightsMatrix))

xAxis = np.linspace(1, len(costGraph), len(costGraph))
plt.plot(xAxis, costGraph, color='red')
plt.ylim(-1, 1)
plt.show()