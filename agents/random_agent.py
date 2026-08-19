"""
agents/random_agent.py — Rastgele ajan.

Aptal, ama işe yarıyor: boru hattının çalıştığını KANITLIYOR.
Rastgele ajan 9 panelde oynayabiliyorsa, sinir ağı da oynayabilir —
çünkü motor ikisi arasındaki farkı bilmiyor.

Sinir ağını yazdığında bunu silmeyeceksin. Karşılaştırma tabanı
olarak kalacak: "eğitilmiş yılan rastgeleden gerçekten iyi mi?"
sorusunun cevabı bu ajanla ölçülür.
"""

import random

from agents.base import Agent
from core import constants as C


class RandomAgent(Agent):
    """Her adımda üç aksiyondan birini rastgele seçer."""

    ACTIONS = (C.GO_FORWARD, C.TURN_RIGHT, C.TURN_LEFT)

    def __init__(self, seed=None, name="random"):
        super().__init__(name=name)

        # Her ajanın KENDİ üreteci. Global random kullanılsaydı
        # 9 ajan birbirinin rastgeleliğini tüketir, davranışları
        # birbirine karışırdı. Ayrıca tohum sayesinde bir oyunu
        # birebir tekrar oynatabilirsin.
        self.rng = random.Random(seed)
        self.seed = seed

    def act(self, state, game=None):
        # state'e bakmıyor. Bakmaması normal — bu ajanın işi
        # akıllı olmak değil, arayüzün çalıştığını göstermek.
        return self.rng.choice(self.ACTIONS)

    def reset(self):
        # Aynı tohumla yeniden başlat: oyun sıfırlandığında ajan da
        # aynı diziyi üretsin. Tekrarlanabilirlik için gerekli.
        self.rng = random.Random(self.seed)


class ForwardAgent(Agent):
    """
    Hep düz giden ajan. Test amaçlı.

    Ne işe yarar: açlık etiketini kontrol etmek, duvara ne kadar
    hızlı çarpıldığını ölçmek, ve "en aptal ajan" tabanı sağlamak.
    Eğitilmiş yılan bundan iyi değilse bir şey ters demektir.
    """

    def __init__(self, name="forward"):
        super().__init__(name=name)

    def act(self, state, game=None):
        return C.GO_FORWARD