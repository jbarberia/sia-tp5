from .optimizer import Adam
from .perceptron_multicapa import PerceptronMulticapa


class Autoencoder(PerceptronMulticapa):
    def __init__(self, layers, bottleneck_index, **kwargs):
        """
        layers: lista de capas (Layer)
        bottleneck_index: índice de la capa de bottleneck (codificación)
        """
        super().__init__(layers, optimizer=Adam(), **kwargs)
        self.bottleneck_index = bottleneck_index

    def encode(self, x):
        """
        Codifica la entrada hasta la capa de bottleneck.
        """
        for layer in self.layers[:self.bottleneck_index + 1]:
            x = layer.forward(x)
        return x

    def decode(self, z):
        """
        Decodifica una representación latente z hasta la salida.
        """
        for layer in self.layers[self.bottleneck_index + 1:]:
            z = layer.forward(z)
        return z

    def reconstruct(self, x):
        """
        Codifica y luego decodifica una entrada.
        """
        return self.batch_forward(x)

    def train_autoencoder(self, x_train, x_out=None, **kwargs):
        """
        Entrena el autoencoder con entrada = salida.
        """
        if x_out is not None:
            return super().train(x_train, x_out, **kwargs)
        else:
            return super().train(x_train, x_train, **kwargs)

