"""
from core.game import Game
g = Game(rows=5, cols=5, seed=1)
print("baş:", g.head, "yön:", g.direction, "yem:", g.food)
print("durum:", g.get_state())
from core import constants as C
for i in range(10):
    print(i, "baş:", g.head, "durum:", g.get_state())
    if not g.is_alive:
        print("öldü:", g.result)
        break
    g.step(C.GO_FORWARD)
"""
"""
from agents.neural_agent import NeuralNetwork
import numpy as np

net = NeuralNetwork(11, 12, 3)

print("genom boyu:", net.genome_size)

v1 = net.to_vector()
net.from_vector(v1)
v2 = net.to_vector()

print("aynı mı:", np.array_equal(v1, v2))
durum = [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]
cikti = net.forward(durum)
print("çıktı:", cikti, "seçilen aksiyon:", np.argmax(cikti))
"""
from agents.neural_agent import NeuralAgent
from core.game import Game

g = Game(rows=10, cols=10, seed=1)
a = NeuralAgent(seed=1)
print("aksiyon:", a.act(g.get_state()))
print("genom boyu:", len(a.genome))