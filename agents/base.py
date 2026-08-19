"""
agents/base.py — Ajan sözleşmesi.

Bu dosyanın tek işi bir kural koymak:
    "Ajana durum verilir, ajan aksiyon döner."

Bu kural sayesinde game.py beynin ne olduğunu HİÇ bilmez.
İnsan, rastgele ajan ve sinir ağı aynı delikten takılıp çıkar.
Sinir ağını eklediğinde motorda tek satır değişmeyecek.
"""

from abc import ABC, abstractmethod


class Agent(ABC):
    """Tüm ajanların uyması gereken sözleşme."""

    def __init__(self, name="agent"):
        self.name = name

    @abstractmethod
    def act(self, state, game=None):
        """
        Durumu al, aksiyon döndür.

        Dönüş: constants.GO_FORWARD / TURN_RIGHT / TURN_LEFT

        game parametresi opsiyonel. Rastgele ajan ve sinir ağı buna
        ihtiyaç duymaz; insan ajanı da duymaz. Ama ileride "tam tahtayı
        gören" bir ajan denemek istersen kapı açık kalsın.
        """
        raise NotImplementedError

    def reset(self):
        """
        Yeni oyun başlarken çağrılır.

        Durumsuz ajanlar için hiçbir şey yapmaz. Ama hafızalı bir ajan
        yazarsan (mesela son N adımı hatırlayan), hafızasını burada
        temizler. Sözleşmede yeri şimdiden dursun.
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name!r}>"