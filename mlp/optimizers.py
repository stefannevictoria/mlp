# SGD Optimizer implementation e outras técnicas de otimização para treinamento do MLP

def sgd(params, grads, lr):
    # Gradient descent: move cada parâmetro na direção oposta ao gradiente
    return [p - lr * g for p, g in zip(params, grads)]