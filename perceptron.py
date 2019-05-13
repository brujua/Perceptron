import sys
import numpy as np
from random import randint
from typing import List
import matplotlib.pyplot as plt

DIMENSION = 2
ERROR_ARGS = "Invalid Argument. \nUsage: python " + sys.argv[0] + " <data-set-file> \
(tab separated, each data point in one line)"
ERROR_NOT_CONVERGED = "Error, training did not converged, the training data must be non linearly separable"
PLOT_TITLE = "Perceptron"
MAX_EPOCHS = 100000
LEARNING_RATE = 0.1

_weights = []


def main(*args):
    file_name = args[0]
    init_weights()
    training_points = parse_training_data(file_name)
    trained = train(training_points)
    if trained:
        draw_plot(training_points, _weights)
    else:
        print(ERROR_NOT_CONVERGED)


def train(training_points: List) -> bool:
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
    return finished


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


def draw_plot(data_points: List, weights: List):
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    for point in data_points:
        plt.plot(point[0], point[1], 'ro' if (point[2] == 1.0) else 'bo')

    slope = -(weights[2] / weights[1]) / (weights[2] / weights[0])
    intercept = -weights[2] / weights[1]
    _draw_line(slope, intercept)
    plt.title(PLOT_TITLE)
    plt.show()


def _draw_line(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


if __name__ == '__main__':
    if len(sys.argv) is not 2:
        print(ERROR_ARGS)
    else:
        main(*sys.argv[1:])
