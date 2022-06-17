# imports
import random
import time
import numpy as np
import math
import MnistDataSet  # this needs to be the file name of the other folder
import copy  # this may cause the problem to

total_time = time.time()  # starts a timer


class i_node:  # input node
    def __init__(self, value, nextlayersize):
        self.weights = [0.0 for i in range(
            nextlayersize)]  # sets a random weight(-1.00, 1.00) for all the synapses spreading out from the input node
        self.value = value


class h_node:  # hidden layer node
    def __init__(self, nextlayersize):
        self.weights = [0.0 for i in range(nextlayersize)]
        self.bias = 0
        self.value = 0


class o_node:  # output node
    def __init__(self, number):
        self.number = number
        self.value = 0  # confidence that the ai has in its decision
        self.bias = 0


NN = []  # this list will store the

weights = []
values = []
bias = []


def init(inputs):
    inputlayer = []
    outputlayer = []
    for i in range(NN_layout[0]):
        inputlayer.append(i_node(inputs[i], NN_layout[1]))
    NN.append(inputlayer)
    for i in range(len(NN_layout) - 2):
        NN.append([h_node(NN_layout[i + 2]) for x in range(NN_layout[i + 1])])
    for i in range(NN_layout[-1]):
        outputlayer.append(o_node(i))
    NN.append(outputlayer)


def adjust_modifiers(NN, variability):
    for i in range(NN_length - 2):
        for node in NN[i + 1]:
            for i, weight in enumerate(node.weights):
                node.weights[i] += round(random.uniform(variability * -1, variability), 2)
            node.bias += round(random.uniform(variability * -1, variability), 2)

    for node in NN[0]:
        for i, weight in enumerate(node.weights):
            node.weights[i] += round(random.uniform(variability * -1, variability), 2)
    for node in NN[-1]:
        node.bias += round(random.uniform(variability * -1, variability), 2)


def display_NN():  # this function is not required for the program to run and is only
    print("inputlayer")
    for node in NN[0]:
        print(node.value, node.weights)
    print("-----------------------------")
    for i in range(len(NN_layout) - 2):
        print("hiddenlayer:", i)
        for node in NN[i + 1]:
            print(i, node.value, node.weights, node.bias)
    print("-----------------------outputlayer-----------------------")
    for node in NN[-1]:
        print(node.number, node.value)
    print("")


def sigmoid(x):  # squashes a number between 0, and 1
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))


def dotproduct(values, weights, bias):
    return np.dot(weights, values) + bias


def cost(expected, actual):
    total = 0
    for i, a in enumerate(expected):
        if a == 1:
            total += ((actual[i] - 1) ** 2)

        elif a == 0:
            total += ((actual[i] - 0) ** 2)

    if expected.index(max(expected)) == actual.index(max(actual)):
        return total, True
    else:
        return total, False


def foward_propagate(set):
    for n in range(NN_length - 1):
        values = np.array([set[n][i].value for i in range(len(set[n]))])
        weights = np.array([set[n][i].weights for i in range(len(set[n]))]).transpose()
        bias = np.array([set[n + 1][i].bias for i in range(len(set[n + 1]))])

        result = dotproduct(values, weights, bias)
        for i, v in enumerate(result):
            set[n + 1][i].value = sigmoid(v)
    output = [set[-1][x].value for x in range(len(set[-1]))]
    fcost, correct = cost(truthtable, output)
    return fcost, correct


generations = 50  # number of generations
iterations = 100  # number of iterations per generation
sample_size = 30  # number of tests for each iteration
NN_layout = [784, 25, 10]  # first number is the size of the input layer, last number is the size of the output layer, all other values are the size of hidden layers
NN_length = len(NN_layout)

given_inputs, label = MnistDataSet.get_image(1)
init(given_inputs)
adjust_modifiers(NN, 1)
best_nn = copy.deepcopy(NN)
set_nn = copy.deepcopy(NN)
set_nn2 = copy.deepcopy(NN)

best_results = 10
set_results = 10
F_results = 0

for i in range(generations):
    # NN is set to the best with adjusted modifiers
    numbers = [random.randint(0, 60000) for y in
               range(sample_size)]  # picks a sample size of random numbers corresponding to the number of

    start_time = time.time()
    best_results = 10
    wins = 0
    for c in range(iterations):
        wins2 = 0
        results = []
        set_nn = copy.deepcopy(set_nn2)
        adjust_modifiers(set_nn, set_results * 100)
        for n, num in enumerate(numbers):
            given_inputs, label = MnistDataSet.get_image(num)
            truthtable = [0 for x in range(10)]
            truthtable[label] = 1

            for a, l in enumerate(set_nn[0]):
                l.value = given_inputs[a]
            # print(set_nn[0][0].weights[0], set_nn[0][300].value)

            res, outcome = foward_propagate(set_nn)
            if outcome:
                wins += 1
                wins2 += 1

            results.append(res)

        accuracy = wins2 / (sample_size) * 100
        F_results = ((sum(results) / len(results))) - accuracy/50

        if F_results < best_results:

            print("secondary score: ", accuracy, "%")
            best_results = F_results
            best_nn = copy.deepcopy(set_nn)

    print(i, ":", best_results)
    print(wins / (sample_size * iterations) * 100, "%")
    set_results = best_results
    set_nn = list(best_nn)
    set_nn2 = list(best_nn)

    generation_time = time.time() - start_time
    print("estimated time: ", (generations - i) * generation_time, i, "% complete")

print("code took: ", time.time() - total_time, "seconds to run")