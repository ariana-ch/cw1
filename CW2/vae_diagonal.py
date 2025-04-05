from CW2.base_vae import VAE
from keras.api.metrics import Mean
from keras import ops
import numpy as np



class VAEDiagonal(VAE):

    def __init__(self, encoder, decoder, num_mc_samples=1, **kwargs):
        """
        You should override this method as necessary in your implementations.
        """
        super().__init__(encoder=encoder, decoder=decoder, num_mc_samples=num_mc_samples, **kwargs)
