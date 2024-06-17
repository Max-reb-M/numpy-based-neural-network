import numpy as np
import copy

matrixMap = [2, 2, 1]
dataSet = [{'input': np.array([0, 1]),'output': np.array([1])},
  {'input': np.array([0, 0]), 'output': np.array([0])}]


def generateWeightsMatrix(neuronsMap):
    num_layers = len(neuronsMap)
    weights = []
    for i in range(num_layers - 1):
        # Добавляем веса для связей между слоями
        #layer_weights = np.random.uniform(low=-3, high=3, size=(neuronsMap[i+1], neuronsMap[i] + 1))
        layer_weights = np.ones((neuronsMap[i+1], neuronsMap[i] + 1))
        #layer_weights = np.zeros((neuronsMap[i+1], neuronsMap[i] + 1), dtype=float)
        weights.append(layer_weights)      
    return weights

weightsMatrix = generateWeightsMatrix(matrixMap)

def activationFunc(activations, weights):
    weightedSum = 0
    for valueIndex in range(len(activations)):
        weightedSum += activations[valueIndex] * weights[valueIndex] # Get the weighted sum
    return 1 / (1 + np.exp(-weightedSum)) # Sigmoid of the weighted sum

def countActivations(inputs, neuronsMap, weights):
    resActivations = [np.array([1, *inputs])] # Add a bias unit to the input layer
    numLayers = len(neuronsMap) # Get the numbers of layers
    for layerInd in range(1, numLayers):
        if layerInd != numLayers - 1: # End layer doesnt have a bias 
            layerActivations = np.array([1])
        else:
            layerActivations = np.array([])
        for neuronInd in range(neuronsMap[layerInd]): # For each unit of the layer
        	layerActivations = np.append(layerActivations, activationFunc(resActivations[layerInd - 1], weights[layerInd - 1][neuronInd])) # Add the according activation(weighted sum) to the layer
        resActivations.append(layerActivations) # Add the layer to the matrix
    return resActivations

def getTheInputOFDataItem(id):
    return dataSet[id]['input']

def getTheOutputOFDataItem(id):
    return dataSet[id]['output']

def costFunctionForOne(inputArr, expectedOutputArr, matrixMap, weights):
    systemOutput = countActivations(inputArr, matrixMap, weights)[-1]
    sumOverExampleErr = 0
    for outputInd in range(len(expectedOutputArr)):
        #sumOverExampleErr += (expectedOutputArr[outputInd] - systemOutput[outputInd])**2 #Square err
        sumOverExampleErr -= (expectedOutputArr[outputInd]*np.log(systemOutput[outputInd]) + (1 - expectedOutputArr)*np.log((1 - systemOutput[outputInd]))) # Logistic err
    return (sumOverExampleErr / len(expectedOutputArr))

def meanCostFunction(dataSet, matrixMap, weights):
    sumOverSetErr = 0
    for dataItemInd in range(len(dataSet)):
        sumOverSetErr += costFunctionForOne(getTheInputOFDataItem(dataItemInd), getTheOutputOFDataItem(dataItemInd), matrixMap, weights)
    return (sumOverSetErr/len(dataSet))

print(weightsMatrix)
print(countActivations(dataSet[0]['input'], matrixMap, weightsMatrix))
print(countActivations(dataSet[1]['input'], matrixMap, weightsMatrix))
print(costFunctionForOne(dataSet[0], matrixMap, weightsMatrix))
print(costFunctionForOne(dataSet[1], matrixMap, weightsMatrix))
print(meanCostFunction(dataSet, matrixMap, weightsMatrix))