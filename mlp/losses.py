# Cross-entropy Loss implementation e outras funções de perda para classificação

import numpy as np

def categorical_cross_entropy(y_true, y_pred):
    # clip evita log(0) → -inf
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))