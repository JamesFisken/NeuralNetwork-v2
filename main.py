import random
import time
import numpy as np
import math
import MnistDataSet

start_time = time.time()

class i_node:
    def __init__(self, value, nextlayersize):
        self.weights = [0.0 for i in range(nextlayersize)]  # sets a random weight(-1.00, 1.00) for all the synapses spreading out from the input node
        self.value = value

class h_node:
    def __init__(self, nextlayersize):
        self.weights = [0.0 for i in range(nextlayersize)]
        self.bias = 0
        self.value = 0

class o_node:
    def __init__(self, number):
        self.number = number
        self.value = 0  # confidence that the ai has in its decision
        self.bias = 0


truthtable = [0, 0, 1]


NN_layout = [4, 3, 3]
NN_length = len(NN_layout)

NN=[]

weights = []
values = []
bias = []
def init(inputs):
    inputlayer = []
    outputlayer = []
    for i in range(NN_layout[0]):
        inputlayer.append(i_node(inputs[i], NN_layout[1]))
    NN.append(inputlayer)
    for i in range(len(NN_layout)-2):
        NN.append([h_node(NN_layout[i+2]) for x in range(NN_layout[i+1])])
    for i in range(NN_layout[-1]):
        outputlayer.append(o_node(i))
    NN.append(outputlayer)

def adjust_modifiers(NN, variability):
    for i in range(NN_length - 2):
        for node in NN[i+1]:
            for i, weight in enumerate(node.weights):
                node.weights[i] += round(random.uniform(variability * -1, variability), 2)
            node.bias += round(random.uniform(variability * -1, variability), 2)

    for node in NN[0]:
        for i, weight in enumerate(node.weights):
            node.weights[i] += round(random.uniform(variability * -1, variability), 2)
    for node in NN[-1]:
        node.bias += round(random.uniform(variability * -1, variability), 2)

def display_NN():
    print("inputlayer")
    for node in NN[0]:
        print(node.value, node.weights)
    print("-----------------------------")
    for i in range(len(NN_layout)-2):
        print("hiddenlayer:", i)
        for node in NN[i+1]:
            print(i, node.value, node.weights, node.bias)
    print("-----------------------outputlayer-----------------------")
    for node in NN[-1]:
        print(node.number, node.value)
    print("")

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dotproduct(values, weights, bias):
    return np.dot(weights, values) + bias

def cost(expected, actual):
    total = 0
    for i, a in enumerate(expected):
        if a == 1:
            total += ((actual[i]-1)**2)
        elif a == 0:
            total += ((actual[i] - 0) ** 2)
    return total
def foward_propagate(NN):
    for n in range(NN_length-1):
        values = np.array([NN[n][i].value for i in range(len(NN[n]))])
        weights = np.array([NN[n][i].weights for i in range(len(NN[n]))]).transpose()
        bias = np.array([NN[n+1][i].bias for i in range(len(NN[n+1]))])

        result = dotproduct(values, weights, bias)
        for i, v in enumerate(result):
            NN[n+1][i].value = sigmoid(v)
    output = [NN[-1][x].value for x in range(len(NN[-1]))]
    fcost = cost(truthtable, output)
    print("cost:", fcost)
    return fcost

given_inputs = [1, 2, 3, 2.5]

init(given_inputs)
adjust_modifiers(NN, 1)
fcost = foward_propagate(NN)
adjust_modifiers(NN, fcost)

foward_propagate(NN)
display_NN()
print("code took: ", time.time() - start_time, "seconds to run")