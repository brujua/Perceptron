import sys
import numpy as np
from random import randint
from typing import List

DIMENSION = 2
ERROR_ARGS = "Invalid Argument. \nUsage: python " + sys.argv[0] + " <data-set-file> \
(tab separated, each data point in one line)"
MAX_EPOCHS = 100000
LEARNING_RATE = 0.1

_weights = []


def main(*args):
    file_name = args[0]
    init_weights()
    training_points = parse_training_data(file_name)
    train(training_points)


def train(training_points: List):
    epoch = 0
    finished = False
    while not finished and epoch < MAX_EPOCHS:
        finished = True
        epoch += 1
        for point in training_points:
            result = evaluate(point[:DIMENSION])
            desired_result = point[DIMENSION]
            pass
            if result != desired_result:
                finished = False
                for i in range(0, DIMENSION):
                    if result < desired_result:
                        _weights[i] += point[i] * LEARNING_RATE
                        _weights[DIMENSION] += LEARNING_RATE  # Bias
                    else:  # result is > than desired
                        _weights[i] -= point[i] * LEARNING_RATE
                        _weights[DIMENSION] -= LEARNING_RATE  # Bias
    if finished:
        print("Trained successful")
    else:
        print("Did not converged")


def evaluate(input_: List) -> int:
    input_.append(1)
    if np.dot(input_, _weights) > 0:
        return 1
    else:
        return 0


def init_weights():
    for _ in range(0, DIMENSION+1):
        _weights.append(randint(-5, 5))


def parse_training_data(file_name: str) -> List:
    training_points = []
    with open(file_name) as file:
        for line in file.readlines():
            data = line.split("\t")
            if len(data) is (DIMENSION + 1):
                point = []
                for i in range(0, len(data)):
                    point.append(int(data[i]))
                training_points.append(point)
    return training_points


if __name__ == '__main__':
    if len(sys.argv) is not 2:
        print(ERROR_ARGS)
    else:
        main(*sys.argv[1:])
