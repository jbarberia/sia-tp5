import numpy as np
from src.perceptron_multicapa import Layer
from src.autoencoder import Autoencoder


def test_autoencoder_decode_uses_all_decoder_layers():
    
    layers = [
        Layer(4, 3, "sigmoid"),
        Layer(3, 2, "sigmoid"),
        Layer(2, 3, "sigmoid"),
        Layer(3, 4, "sigmoid") 
    ]

    autoencoder = Autoencoder(layers, bottleneck_index=1)

    z = np.array([0.5, -0.3])
    autoencoder.decode(z)
    assert len(autoencoder.layers) == 4
    
    z = np.array([0.5, -0.3, 1.0, 2.0])
    autoencoder.encode(z)
    assert len(autoencoder.layers) == 4
