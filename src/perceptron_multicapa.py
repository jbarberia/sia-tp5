from dataclasses import dataclass
import pickle
import numpy as np
from time import time
from .optimizer import SGD

def sigmoid(x):
    # Numerically stable version
    return np.where(x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)))
    
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def cosine(x):
    return np.cos(x)

def cosine_derivative(x):
    return - np.sin(x)

def relu(x):
    return np.where(x >= 0, x, -0.01 * x)

def relu_derivative(x):
    return np.where(x >= 0, 1, 0.01)


FUNCTIONS = {
    "sigmoid": sigmoid,
    "linear": linear,
    "cosine": cosine,
    "relu": relu
}

DERIVATIVES = {
    "sigmoid": sigmoid_derivative,
    "linear": linear_derivative,
    "cosine": cosine_derivative,
    "relu": relu_derivative,
}

class Layer:
    def __init__(self, dims_in, dims_out, activation_function="sigmoid"):
        self.w = np.random.randn(dims_out, dims_in)
        self.b = np.zeros(dims_out)
        self.activation = FUNCTIONS[activation_function]
        self.activation_derivative = DERIVATIVES[activation_function]
        self.x = None
        self.z = None
        self.a = None
        self.dims_in = dims_in
        self.dims_out = dims_out

    def forward(self, x):
        self.x = x
        self.z = self.w @ x + self.b
        self.a = self.activation(self.z)
        return self.a
    
    def batch_forward(self, X):        
        Z = X @ self.w.T + self.b
        A = self.activation(Z)
        return A
    
    def backward(self, grad_out):
        # https://en.wikipedia.org/wiki/Backpropagation
        # delta_j = dz = dE/do * do/dnet = grad_out * diff_activation_fun
        # dWij = -n * delta_j * out_i = -n * dz .* x
        dz = grad_out * self.activation_derivative(self.z) # delta = dL/d0 * dphi/dnet
        dw = np.outer(dz, self.x)  
        db = dz                    
        grad_input = self.w.T @ dz # sum(w * dl) * dz
        return grad_input, dw, db


class StochasticLayer(Layer):
    def __init__(self, dims_in, latent_dim):
        super().__init__(dims_in, 2 * latent_dim, activation_function="linear")
        self.latent_dim = latent_dim
        self.mu = None
        self.log_var = None


    def forward(self, x):
        self.x = x
        self.z_all = self.w @ x + self.b
        
        self.mu = self.z_all[:self.latent_dim]
        self.log_var = self.z_all[self.latent_dim:]

        # Reparameterization trick - tomo un valor muestreado normal
        self.eps = np.random.randn(*self.mu.shape)
        std = np.exp(0.5 * self.log_var)
        return self.mu + std * self.eps
        

    def batch_forward(self, X):
        Z_all = X @ self.w.T + self.b

        mu = Z_all[:, :self.latent_dim]
        log_var = Z_all[:, self.latent_dim:]
        
        eps = np.random.randn(*mu.shape)
        std = np.exp(0.5 * log_var)
        return mu + std * eps


    def backward(self, grad_out):
        std = np.exp(0.5 * self.log_var)
        d_mu = 1
        d_logvar = 0.5 * std * self.eps

        dL_d_mu = grad_out * d_mu
        dL_d_logvar = grad_out * d_logvar

        dz = np.concatenate([dL_d_mu, dL_d_logvar])
        dw = np.outer(dz, self.x)
        db = dz
        grad_input = self.w.T @ dz

        return grad_input, dw, db


class PerceptronMulticapa:
    def __init__(self, layers, optimizer=None, loss_fun="mse"):
        self.layers = layers
        self.optimizer = optimizer if optimizer else SGD()
        self.idx2class = None
        self.loss_fun = loss_fun
        

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    

    def batch_forward(self, X):
        for layer in self.layers:
            X = layer.batch_forward(X)
        return X
    

    def backward(self, x, y):        
        prediction = self.forward(x)
        grad = prediction - y
        for layer in reversed(self.layers):
            grad, dw, db = layer.backward(grad)
            self.optimizer.update(layer, dw, db)
        return 0.5 * np.sum((prediction - y) ** 2)
    

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=1000, batch_size=1, k_fold=None):
        if k_fold:
            X = np.vstack((x_train, x_val)) if x_val else x_train
            Y = np.vstack((y_train, y_val)) if y_val else y_train
            #indices = [[i for i in range(X.shape[0])] for k in range(k_fold)]
            n_samples = len(X)
            n_fold = n_samples // k_fold

            from itertools import cycle
            indices = np.arange(0, k_fold * n_fold)
            idx_val_cycle = cycle(np.arange(k * n_fold, (k+1) * n_fold) for k in range(k_fold))

        history = []
        elapsed_time = 0
        for epoch in range(epochs):
            t0 = time()
            if k_fold:
                idx_val = next(idx_val_cycle)
                idx_train = [x for x in indices if x not in idx_val]

                x_train = X[idx_train]
                y_train = Y[idx_train]
                x_val = X[idx_val]
                y_val = Y[idx_val]
        

            batch_loss = self._train_model(x_train, y_train, batch_size, epoch)
                
            t1 = time()
            elapsed_time += t1 - t0

            # entrenamiento
            y_hat_train = self.batch_forward(x_train)
            train_loss = np.mean((y_hat_train - y_train) ** 2)

            # validacion
            val_loss = None
            if x_val is not None and y_val is not None:
                y_hat_val = self.batch_forward(x_val)
                val_loss = 0.5 * np.mean((y_hat_val - y_val) ** 2)

            history.append({
                "epoch": epoch,
                "training_time": t1 - t0,
                "elapsed_training_time": elapsed_time,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

        return history
    

    def _train_model(self, X, Y, batch_size, epoch=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        total_loss = 0
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for i in range(n_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
        
            batch_X = X[batch_indices]
            batch_Y = Y[batch_indices]
        
            # 1 - Forward pass
            batch_loss = 0.0
            grads_dw = [np.zeros_like(layer.w) for layer in self.layers]
            grads_db = [np.zeros_like(layer.b) for layer in self.layers]
            for x, y in zip(batch_X, batch_Y):
                y_hat = self.forward(x)
                
                loss, grad = self._loss_and_grad(y, y_hat)
                batch_loss += loss

                # 2 - Calculate weights and bias grad
                local_grads = []
                for layer in reversed(self.layers):
                    grad, dw, db = layer.backward(grad)
                    local_grads.append((dw, db))

                for idx, (dw, db) in enumerate(reversed(local_grads)):
                    grads_dw[idx] += dw
                    grads_db[idx] += db

            # 3 - Backward pass
            for layer, dw_sum, db_sum in zip(self.layers, grads_dw, grads_db):
                self.optimizer.update(layer, dw_sum / len(batch_X), db_sum / len(batch_X))

            total_loss += batch_loss

        return total_loss / n_samples


    def _loss_and_grad(self, y, y_hat):
        if self.loss_fun == "mse":
            fun = 0.5 * np.mean((y_hat - y)**2)
            grad = y_hat - y
        
        elif self.loss_fun == "bce":
            epsilon = 1e-6
            y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
            
            fun = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            grad = -(y / y_hat - (1 - y) / (1 - y_hat))
            
        return fun, grad


    def predict(self, x):
        pred = []
        for xi in x:
            forw = self.forward(xi)
            indx = np.argmax(forw)
            if self.idx2class:
                pred.append(self.idx2class[indx])
            else:
                pred.append(indx)
        return pred


    def one_hot_encoding(self, y: np.ndarray):
        val2idx = {val: i for i, val in enumerate(np.unique(y))}
        idx2val = {i: val for i, val in enumerate(np.unique(y))}
        dims = y.shape[0], len(val2idx)
        output = np.zeros(dims)

        for i, y_i in enumerate(y):
            output[i, val2idx[y_i]] = 1
        
        self.idx2class = idx2val
        return output


    def one_hot_decoding(self, x):
        if self.idx2class:
            return self.idx2class[np.argmax(x)]
        else:
            return np.argmax(x)


    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)


    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
