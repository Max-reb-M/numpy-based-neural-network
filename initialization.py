import numpy as np
import copy

inputSample = [0, 1]
outputSample = [1]

def generateWeightsMatrix(neuronsMap):
    num_layers = len(neuronsMap)
    weights = []
    for i in range(num_layers - 1):
        # Добавляем веса для связей между слоями
        layer_weights = np.random.uniform(low=-3, high=3, size=(neuronsMap[i+1], neuronsMap[i] + 1))
        #layer_weights = np.ones((neuronsMap[i+1], neuronsMap[i] + 1))
        #layer_weights = np.zeros((neuronsMap[i+1], neuronsMap[i] + 1), dtype=float)
        weights.append(layer_weights)      
    return weights

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

# Пример использования:
neuronsMapSample = [2, 1]
# Например, 3 нейрона во входном слое, 4 в скрытом, 2 в выходном
weightsMatrixSample = generateWeightsMatrix(neuronsMapSample)
activationsMatrixSample = countActivations(inputSample, neuronsMapSample, weightsMatrixSample)
print('Weights:   ', weightsMatrixSample)
print('Activations:   ', activationsMatrixSample)

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

def sumOverArr(arr):
    res = 0
    for i in arr:
        res += i
    return res

def sumOverTwoDimMatrix(matrix):
    res = 0
    for i in matrix:
        for j in i:
            res += j
    return res

def buildGradient(neuronsMap, weights, inputArr, expectedOutputArr):
    gradientsMatrix = copy.deepcopy(weights) # lazy way to get the same shaped matrix
    #weightsTempMatrix[layerInd][rightNeuronIndex][leftNeuronIndex] += (1 / accurancyRate)
    activationsTempMatrix = countActivations(inputArr, neuronsMap, gradientsMatrix)
    #print('Inside the function', costFunction(activationsTempMatrix, expectedOutput) - costFunction(activationsMatrix, expectedOutput))
    #return ((costFunction(dataSet, neuronsMap, weightsTempMatrix) - costFunction(dataSet, neuronsMap, weights)) * accurancyRate)
    systemOutput = activationsTempMatrix[-1].copy()
    localGradientForCurrentLayer = 0
    for outputInd in range(len(expectedOutputArr)):
            localGradientForCurrentLayer += (-expectedOutputArr[outputInd] * (1 / systemOutput[outputInd]) + (1 - expectedOutputArr[outputInd])/(1 - systemOutput[outputInd])) * (systemOutput[outputInd]) * (1 - systemOutput[outputInd])
    localGradientForCurrentLayer = localGradientForCurrentLayer / len(expectedOutputArr)
    for layerInd in range(-1, -len(neuronsMap) -1, -1):
        for rightNeuronConnectionInd in range(1, neuronsMap[layerInd]):
            print(layerInd, neuronsMap)
            for leftNeuronConnectionInd in range(neuronsMap[layerInd - 1]):
                gradientsMatrix[layerInd][rightNeuronConnectionInd][leftNeuronConnectionInd] = localGradientForCurrentLayer * activationsTempMatrix[layerInd - 1][leftNeuronConnectionInd]
        localGradientForCurrentLayer = localGradientForCurrentLayer * sumOverArr(activationsTempMatrix[layerInd - 1]) * sumOverTwoDimMatrix(weights[layerInd])
    


#print('Cost conflict:   ', costFunction(activationsMatrixSample, [1]))
#print('Cost der:   ', costFuncDerivative(1, 0, 1, weightsMatrixSample, activationsMatrixSample, neuronsMapSample, inputSample, outputSample, 1000))

def getMeanGradient(gradientsMatrix):
    meanGradient = copy.deepcopy(gradientsMatrix[0]) # lazy way to get the same shaped matrix
    tempSum = 0
    for layerInd in range(len(meanGradient)):
        for rightNeuronConnectionInd in range(len(meanGradient[layerInd])):
            for leftNeuronConnectionInd in range(len(meanGradient[layerInd][rightNeuronConnectionInd])):
                for gradient in gradientsMatrix:
                    tempSum += gradient[layerInd][rightNeuronConnectionInd][leftNeuronConnectionInd]
                    #print('83 getMeanGradient:   ', tempSum)
                meanGradient[layerInd][rightNeuronConnectionInd][leftNeuronConnectionInd] = tempSum / len(gradientsMatrix)
                tempSum = 0
    return meanGradient