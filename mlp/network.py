# Implementação do MLP

import numpy as np
from mlp.activations import relu, relu_derivative, softmax
from mlp.losses import categorical_cross_entropy
from mlp.optimizers import sgd


def init_params(layer_sizes, seed=42):
    """
    Inicializa pesos com He initialization e biases com zero.
    layer_sizes: lista com o tamanho de cada camada, ex: [784, 128, 64, 10]
    """
    np.random.seed(seed)
    params = []
    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]
        W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
        b = np.zeros((1, n_out))
        params.extend([W, b])
    return params


def forward(X, params):
    """
    Forward pass para arquitetura com número arbitrário de camadas ocultas.
    Camadas ocultas usam ReLU, camada de saída usa Softmax.
    Retorna a predição final e o cache com z e a de cada camada.
    """
    cache = []
    n_layers = len(params) // 2  # cada camada tem W e b

    a = X
    for i in range(n_layers):
        W = params[i * 2]
        b = params[i * 2 + 1]
        z = a @ W + b

        # Última camada: softmax. Demais: ReLU
        if i == n_layers - 1:
            a = softmax(z)
        else:
            a = relu(z)

        cache.append((z, a))

    return a, cache


def backward(X, y_true, params, cache):
    """
    Backpropagation com regra da cadeia, de trás pra frente.
    Na saída usa a derivada simplificada softmax + cross-entropy: ŷ - y
    Nas camadas ocultas propaga o gradiente através da derivada da ReLU.
    """
    n = X.shape[0]
    n_layers = len(params) // 2
    grads = [None] * len(params)

    # Gradiente da camada de saída
    _, a_out = cache[-1]
    dz = (a_out - y_true) / n

    for i in reversed(range(n_layers)):
        W = params[i * 2]
        # Ativação da camada anterior (ou X se for a primeira camada)
        a_prev = cache[i - 1][1] if i > 0 else X

        grads[i * 2]     = a_prev.T @ dz
        grads[i * 2 + 1] = np.sum(dz, axis=0, keepdims=True)

        if i > 0:
            z_prev = cache[i - 1][0]
            dz = (dz @ W.T) * relu_derivative(z_prev)

    return grads


def train(X, y, layer_sizes, epochs=10, lr=0.1, batch_size=64, seed=42):
    params = init_params(layer_sizes, seed)
    loss_history = []
    acc_history = []
    n = X.shape[0]

    for epoch in range(epochs):
        # Embaralha os dados a cada época para evitar viés de ordem
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X[idx], y[idx]

        epoch_loss = 0
        num_batches = 0

        for i in range(0, n, batch_size):
            X_batch = X_shuf[i:i + batch_size]
            y_batch = y_shuf[i:i + batch_size]

            a_out, cache = forward(X_batch, params)
            loss = categorical_cross_entropy(y_batch, a_out)
            epoch_loss += loss
            num_batches += 1

            grads = backward(X_batch, y_batch, params, cache)
            params = sgd(params, grads, lr)

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        a_full, _ = forward(X, params)
        preds = np.argmax(a_full, axis=1)
        labels = np.argmax(y, axis=1)
        acc = np.mean(preds == labels)
        acc_history.append(acc)

        print(f"Época {epoch+1:2d} | Loss: {avg_loss:.4f} | Acurácia treino: {acc*100:.1f}%")

    return params, loss_history, acc_history