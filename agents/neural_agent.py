import numpy as np
from agents.base import Agent
from core import constants as C
import config


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, rng=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        rng = rng or np.random.default_rng()

        self.w1 = rng.normal(0, 1, (input_size, hidden_size))
        self.b1 = rng.normal(0, 1, hidden_size)
        self.w2 = rng.normal(0, 1, (hidden_size, output_size))
        self.b2 = rng.normal(0, 1, output_size)

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        hidden = np.tanh(x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    def forward_debug(self, x):
        """
        forward ile ayni hesap, ama ARA DEGERLERI de dondurur.
        """
        x = np.asarray(x, dtype=float)
        hidden = np.tanh(x @ self.w1 + self.b1)
        out = hidden @ self.w2 + self.b2
        return x, hidden, out

    @property
    def genome_size(self):
        return (
            self.input_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.output_size
            + self.output_size
        )

    def to_vector(self):
        vector = [self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), self.b2.ravel()]
        return np.concatenate(vector)

    def from_vector(self, vector):
        vector = np.asarray(vector, dtype=float)
        if vector.size != self.genome_size:
            raise ValueError(...)
        i = 0
        n = self.input_size * self.hidden_size
        self.w1 = vector[i : i + n].reshape(self.input_size, self.hidden_size)
        i += n
        n = self.hidden_size
        self.b1 = vector[i : i + n].copy()
        i += n
        n = self.hidden_size * self.output_size
        self.w2 = vector[i : i + n].reshape(self.hidden_size, self.output_size)
        i += n
        n = self.output_size
        self.b2 = vector[i : i + n].copy()
        return self


class NeuralAgent(Agent):

    def __init__(self, genome=None, seed=None, name="neural"):
        super().__init__(name=name)
        rng = np.random.default_rng(seed)
        self.net = NeuralNetwork(config.STATE_SIZE, config.HIDDEN_SIZE, C.ACTION_NUM, rng=rng)
        if genome is not None:
            self.net.from_vector(genome)

    def act(self, state, game=None):
        scores = self.net.forward(state)
        index = np.argmax(scores)
        return int(index)

    @property
    def genome(self):
        return self.net.to_vector()

    @genome.setter
    def genome(self, vector):
        self.net.from_vector(vector)
