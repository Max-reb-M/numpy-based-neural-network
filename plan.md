# 2 Files:
    - initialization file with all functions
    - main file with implementation

## main file:
    1. Prepare Dataset
    2. Ask for the number of layers and number of units in them
    3. Create a matrix of random weights (`generateWeightsMatrix`)
    4. (?) Count the matrix of units, and find the value od Cost Function 'countActivations', 'costFunctionForOne' and 'meanCostFunction'
    5. REPEAT ---> Reduce the Cost Function by changing the weights WHILE delta between iterations is lower than given ´convergance rate´ and the number of iterations is lower than given limit:
        - 5.1. Find a list of gradients matrix using ´buildGradient´ method
        - 5.2. Find a mean gradient matrix using ´modelLearning´ method
        - 5.3. Change the weights in order to do a gradient decsent
    6. Show the graphics of: (Train error / number of iterations)

## initialization file methods:

```python
    generateWeightsMatrix( neuronsMap ):
        return weightsMatrix
    
    countActivations( inputs , neuronsMap, weights):
        return activations

    costFunctionForOne(dataItem, matrixMap, weights):
        return activations
    
    meanCostFunction(dataSet, matrixMap, weights):
        costFunctionForOne
        return activations

    buildGradient(neuronsMap, weights, inputArr, expectedOutput):
        countActivations(inputArr, neuronsMap, gradientsMatrix)

    
```

## in main file:
```python
        for dataItem in dataSet:
            allGradients.append(buildGradient(matrixMap, weights, dataItem['input'], dataItem['output']))
        weightsGradient = getMeanGradient(allGradients)
```
