import numpy as np
import matplotlib as mpl


def u(W, X, theta):
    return np.dot(W, X) - theta


def classify(W, X, theta):
    return 1 if u(W, X, theta) >= 0 else 0


def RNA(low, high, size, epochs):
    resultados = np.array([])
    classes = np.array([])
    theta = 0
    for epoch in range(epochs):
        W = np.random.uniform(low=low, high=high, size=size)
        X = np.random.uniform(low=low, high=high, size=size)
        resultado = u(W, X, theta)
        resultados = np.append(resultados, resultado)
        classe = classify(W, X, theta)
        classes = np.append(classes, classe)
    print(resultados, classes)


RNA(-5.5, 5.5, 3, 10)
