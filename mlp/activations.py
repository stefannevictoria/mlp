# Funções de ativação e suas derivadas

import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    # Retorna 1 onde o neurônio foi ativado, 0 onde foi zerado pela ReLU
    return (x > 0).astype(float)

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    # Subtrai o máximo por estabilidade numérica: evita overflow no exp()
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)