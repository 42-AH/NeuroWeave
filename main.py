# When using NeuroWeave, please give credit.
# https://replit.com/@42AH/NeuroWeave


import math
import random


def make_nn(input_size, hidden_size, hidden2_size, hidden3_size, output_size):
    global inputs, hiddens1, hiddens2, hiddens3, outputs
    global weight1, weight2, weight3, weight4
    global bias1, bias2, bias3, bias4

    inputs = [random.random() for _ in range(input_size)]
    hiddens1 = [0] * hidden_size
    hiddens2 = [0] * hidden2_size
    hiddens3 = [0] * hidden3_size
    outputs = [0] * output_size

    weight1 = [random.random() for _ in range(input_size * hidden_size)]
    weight2 = [random.random() for _ in range(hidden_size * hidden2_size)]
    weight3 = [random.random() for _ in range(hidden2_size * hidden3_size)]
    weight4 = [random.random() for _ in range(hidden3_size * output_size)]

    bias1 = [random.random() for _ in range(hidden_size)]
    bias2 = [random.random() for _ in range(hidden2_size)]
    bias3 = [random.random() for _ in range(hidden3_size)]
    bias4 = [random.random() for _ in range(output_size)]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forwardpropagate(dropout_prob):
    for i in range(len(hiddens1)):
        hiddens1[i] = sigmoid(sum(inputs[i2] * weight1[i * len(inputs) + i2] for i2 in range(len(inputs))) + bias1[i])
        hiddens1[i] = hiddens1[i] * (random.random() < dropout_prob)
    for i in range(len(hiddens2)):
        hiddens2[i] = sigmoid(sum(hiddens1[i2] * weight2[i * len(hiddens1) + i2] for i2 in range(len(hiddens1))) + bias2[i])
        hiddens2[i] = hiddens2[i] * (random.random() < dropout_prob)
    for i in range(len(hiddens3)):
        hiddens3[i] = sigmoid(sum(hiddens2[i2] * weight3[i * len(hiddens2) + i2] for i2 in range(len(hiddens2))) + bias3[i])
        hiddens3[i] = hiddens3[i] * (random.random() < dropout_prob)
    for i in range(len(outputs)):
        outputs[i] = sigmoid(sum(hiddens3[i2] * weight4[i * len(hiddens3) + i2] for i2 in range(len(hiddens3))) + bias4[i])

def backpropagate():
    global actual
    global error
    global bias1, bias2, bias3, bias4

    error = actual - outputs[0]
    gradient4 = -2 * error * outputs[0] * (1 - outputs[0])
    gradient3 = [gradient4 * weight4[i] * hiddens3[i] * (1 - hiddens3[i]) for i in range(len(hiddens3))]
    gradient2 = [sum(gradient3[i2] * weight3[i * len(hiddens3) + i2] * hiddens2[i] * (1 - hiddens2[i]) for i2 in range(len(hiddens3))) for i in range(len(hiddens2))]
    gradient1 = [sum(gradient2[i2] * weight2[i * len(hiddens2) + i2] * hiddens1[i] * (1 - hiddens1[i]) for i2 in range(len(hiddens2))) for i in range(len(hiddens1))]

    learning_rate = 0.1

    for i in range(len(weight1)):
        weight1[i] -= learning_rate * gradient1[i % len(hiddens1)] * inputs[i // len(hiddens1)]
    for i in range(len(weight2)):
        weight2[i] -= learning_rate * gradient2[i % len(hiddens2)] * hiddens1[i // len(hiddens2)]
    for i in range(len(weight3)):
        weight3[i] -= learning_rate * gradient3[i % len(hiddens3)] * hiddens2[i // len(hiddens3)]
    for i in range(len(weight4)):
        weight4[i] -= learning_rate * gradient4 * hiddens3[i // len(outputs)]

    bias1 = [bias1[i] - learning_rate * gradient1[i] for i in range(len(bias1))]
    bias2 = [bias2[i] - learning_rate * gradient2[i] for i in range(len(bias2))]
    bias3 = [bias3[i] - learning_rate * gradient3[i] for i in range(len(bias3))]
    bias4 = [bias4[i] - learning_rate * gradient4 for i in range(len(bias4))]

    return error

make_nn(2, 2, 3, 2, 1)

inputs[0] = random.uniform(0.0, 0.5)
inputs[1] = random.uniform(0.0, 0.5)
actual = inputs[0] + inputs[1]

iterations = int(input("Iterations: "))
current = 0
while current < iterations:
    forwardpropagate(0.025)
    print(backpropagate())
    if abs(actual - outputs[0]) < 0.001:
      inputs[0] = random.uniform(0.0, 0.5)
      inputs[1] = random.uniform(0.0, 0.5)
      actual = inputs[0] + inputs[1]
      current += 1
num1 = -1
num2 = -1
while not 0 <= num1 <= 1:
  num1 = float(input("Number 1 (0.0-0.5): "))
while not 0 <= num2 <= 1:
  num2 = float(input("Number 2 (0.0-0.5): "))
actual = num1 + num2
inputs[0] = num1
inputs[1] = num2
forwardpropagate(-1)
print("AI: " + str(outputs[0]))
print("Error: " + str(outputs[0] - actual))
