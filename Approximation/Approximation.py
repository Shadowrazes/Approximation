import numpy as np
import matplotlib.pyplot as plt

M = 2

class Point():
    def __init__(self, x, y):
        self.X = x
        self.Y = y

Points = [Point(0.5, 0.5), Point(0.7, 0.3), Point(0.9, 1), Point(1, 0.8), Point(1.3, 2), Point(1.45, 1.8)]

def sortMatrix(matrix, mLen):
    rowMaxElem = 0
    for i in range(mLen):
        for j in range(i + 1, mLen):
            column = 0
            while matrix[i][column] == matrix[j][column]:
                column += 1
                if column == mLen + 1:
                    return
            if abs(matrix[i][column]) < abs(matrix[j][column]):
                matrix[:, [i, j]] = matrix[:, [j, i]]

def printMatrix(matrix, mLen):
    for i in range(mLen):
        print(matrix[i])

def gauss(matrix, mLen):

    print("Исходная матрица:")
    printMatrix(matrix, mLen)

    sortMatrix(matrix, mLen)

    for step in range(mLen - 1):
        for row in range(step + 1, mLen):
            if (matrix[step][step] == 0):
                continue
            coefficient = matrix[row][step] / -matrix[step][step]
            for column in range(mLen + 1):
                matrix[row][column] += round(coefficient * matrix[step][column], 10)
        sortMatrix(matrix, mLen)

    step = 0
    results = []
    for row in range(mLen - 1, -1, -1):
        count = 0
        for column in range(mLen - 1, mLen - step - 1, -1):
            matrix[row][mLen] -= matrix[row][column] * results[count]
            count += 1
        results.append(round(matrix[row][mLen] / matrix[row][mLen - step - 1], 3))
        step += 1

    return np.array(results)

def sumX(step):
    sum = 0
    for i in range(len(Points)):
        sum += Points[i].X ** step
    return sum

def sumY(step):
    sum = 0
    for i in range(len(Points)):
        sum += Points[i].Y * (Points[i].X ** step)
    return sum

def initMatrix():
    matrix = np.empty((M + 1, M + 2))
    for i in range(M + 1):
        for j in range(M + 1):
            matrix[i][j] = sumX(i + j)
        matrix[i][len(matrix[i]) - 1] = sumY(i)
    return matrix

def graph(X, Y, descr, figureNum):
    plt.figure(figureNum)
    plt.plot(X, Y, label = descr)
    plt.legend()

def approxFunc(roots, x):
    sum = 0
    for i in range(len(roots)):
        sum += roots[i] * (x ** i)
    return sum

matrix = initMatrix()
roots = gauss(matrix, len(matrix))
print("\nКорни", roots)

interval = np.arange(0, 2, 0.1)
funcVal = []

for i in range(len(interval)):
    funcVal.append(approxFunc(roots, interval[i]))
    
graph(interval, funcVal, "~x^2", 1)
for i in range(len(Points)):
    plt.scatter(Points[i].X, Points[i].Y)
plt.show()
   