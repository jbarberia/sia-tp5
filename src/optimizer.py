import numpy as np

class Optimizer:
    def update(self, layer, dw, db):
        return NotImplemented


class SGD(Optimizer):
    learning_rate = 0.01
    
    def update(self, layer, dw, db):
        layer.w -= self.learning_rate * dw
        layer.b -= self.learning_rate * db


class Momentum(Optimizer):
    learning_rate = 0.01
    momentum = 0.8
    
    def update(self, layer, dw, db):
        prev_update_w = layer.__dict__.get("prev_update_w", 0)
        prev_update_b = layer.__dict__.get("prev_update_b", 0)

        delta_w = self.learning_rate * dw + self.momentum * prev_update_w
        delta_b = self.learning_rate * db + self.momentum * prev_update_b
        
        layer.prev_update_w = delta_w
        layer.prev_update_b = delta_b
        
        layer.w -= delta_w
        layer.b -= delta_b


class AdaGrad(Optimizer):
    learning_rate = 0.01
    epsilon = 1e-8

    def update(self, layer, dw, db):
        velocity_w = layer.__dict__.get("velocity_w", 0)
        velocity_b = layer.__dict__.get("velocity_b", 0)

        layer.velocity_w = velocity_w + dw**2
        layer.velocity_b = velocity_b + db**2

        layer.w -= self.learning_rate / (np.sqrt(layer.velocity_w) + self.epsilon) * dw
        layer.b -= self.learning_rate / (np.sqrt(layer.velocity_b) + self.epsilon) * db


class RMSprop(Optimizer):
    learning_rate = 0.01
    decay_rate = 0.999
    epsilon = 1e-8

    def update(self, layer, dw, db):
        velocity_w = layer.__dict__.get("velocity_w", 0)
        velocity_b = layer.__dict__.get("velocity_b", 0) 

        layer.velocity_w = self.decay_rate * velocity_w + (1 - self.decay_rate) * dw**2
        layer.velocity_b = self.decay_rate * velocity_b + (1 - self.decay_rate) * db**2

        layer.w -= self.learning_rate / (np.sqrt(layer.velocity_w) + self.epsilon) * dw
        layer.b -= self.learning_rate / (np.sqrt(layer.velocity_b) + self.epsilon) * db


class Adam(Optimizer):
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    def update(self, layer, dw, db):
        t = layer.__dict__.get("t", 0) + 1
        layer.t = t

        momentum_w = layer.__dict__.get("momentum_w", 0)
        momentum_b = layer.__dict__.get("momentum_b", 0)

        velocity_w = layer.__dict__.get("velocity_w", 0)
        velocity_b = layer.__dict__.get("velocity_b", 0)

        layer.momentum_w = self.beta1 * momentum_w + (1 - self.beta1) * dw
        layer.momentum_b = self.beta1 * momentum_b + (1 - self.beta1) * db

        layer.velocity_w = self.beta2 * velocity_w + (1 - self.beta2) * dw**2
        layer.velocity_b = self.beta2 * velocity_b + (1 - self.beta2) * db**2
       
        momentum_w_hat = layer.momentum_w / (1 - self.beta1**(t))
        momentum_b_hat = layer.momentum_b / (1 - self.beta1**(t))

        velocity_w_hat = layer.velocity_w / (1 - self.beta2**(t))
        velocity_b_hat = layer.velocity_b / (1 - self.beta2**(t))

        layer.w -= self.learning_rate * momentum_w_hat / (np.sqrt(velocity_w_hat) + self.epsilon)
        layer.b -= self.learning_rate * momentum_b_hat / (np.sqrt(velocity_b_hat) + self.epsilon)
