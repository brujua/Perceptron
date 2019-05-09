import sys
from random import randint
from typing import List

DIMENSION = 3
ERROR_ARGS = "Invalid Argument. \nUsage: python " + sys.argv[0] + " <data-set-file> \
(tab separated, each data point in one line)"
weights = []


def train(training_points: List):
    pass


def main(*args):
    file_name = args[0]
    init_weights()
    training_points = parse_training_data(file_name)
    train(training_points)


def init_weights():
    for i in range(0, DIMENSION):
        weights[i] = randint(-5, 5)


def parse_training_data(file_name: str) -> List:
    training_points = []
    with open(file_name) as file:
        for line in file.readlines():
            data = line.split("\t")
            if len(data) is (DIMENSION + 1):
                training_points.append(data)
    return training_points


if __name__ == '__main__':
    if len(sys.argv) is not 2:
        print(ERROR_ARGS)
    else:
        main(*sys.argv[1:])
